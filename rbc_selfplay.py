#!/usr/bin/env python
# coding: utf-8

# # RBC AlphaZero-like Bot
# 
# Reconnaissance Blind Chess (RBC) is a partially observable variant of chess in which players have perfect information about their own pieces but only limited observations of the opponent.
# 
# This notebook describes a learning-based RBC agent inspired by the AlphaZero framework, combining neural network evaluation, search, and self-play training. The emphasis is on a clear and rule-compliant implementation that can be trained and evaluated end to end.

# ##DEPENDENCIES
# 

# This section lists the external libraries required to run the notebook.
# 
# The implementation relies on standard numerical and deep learning tools for tensor computation and optimization, together with a chess engine library and the ReconChess framework to ensure correct handling of game rules and interaction between agents.
# 
# These dependencies provide the basic infrastructure for representing game states, running self-play matches, training neural networks, and evaluating the resulting agent.

# In[ ]:


# %pip -q install python-chess reconchess



# Install required dependencies.
#  - python-chess: standard chess representation and move generation
#  - reconchess: official framework for Reconnaissance Blind Chess,
#    enabling rule-compliant gameplay against other bots
# 

# In[ ]:


import os # File system utilities (paths, directories, checkpoints/logging)
import math
import  random # Lightweight randomness (fallback move selection, seeding)

from dataclasses import dataclass # Convenience for configuration structs
from typing import Dict, Tuple, List, Optional, Any

import numpy as np # Fast numeric arrays for buffers, targets, and logging
import torch # Core deep learning framework (tensors, GPU support, training)
import torch.nn as nn  # Neural network layers/modules (CNN trunk, policy/value heads)
import torch.nn.functional as F # Losses and activations (KL, MSE, softmax, ReLU)

import chess
import reconchess  # RBC framework (game loop, sensing, move resolution, opponents)
from reconchess import Player, Color, Square

import csv # Experiment logging to CSV
import datetime # Timestamps for experiment logs and checkpoints




# RANDOMBOT

import random
from reconchess import Player

class SimpleRandomBot(Player):
    def handle_game_start(self, color, board, opponent_name):
        self.color = color

    def handle_opponent_move_result(self, captured_my_piece, capture_square):
        pass

    def choose_sense(self, sense_actions, move_actions, seconds_left):
        return random.choice(sense_actions)

    def handle_sense_result(self, sense_result):
        pass

    def choose_move(self, move_actions, seconds_left):
        legal = [m for m in move_actions if m is not None]
        return random.choice(legal) if legal else None

    def handle_move_result(self, requested_move, taken_move, captured_opponent_piece, capture_square):
        pass

    def handle_game_end(self, winner_color, win_reason, game_history):
        pass

# ===============================
# INTERNAL ELO (inline, no imports)
# ===============================
from dataclasses import dataclass
from typing import Dict, Tuple, Callable

@dataclass
class EloConfig:
    k: float = 32.0
    scale: float = 400.0
    base: float = 10.0

def expected_score(r_a: float, r_b: float, cfg: EloConfig) -> float:
    return 1.0 / (1.0 + cfg.base ** ((r_b - r_a) / cfg.scale))

def score_from_counts(wins: int, losses: int, draws: int) -> float:
    total = wins + losses + draws
    if total <= 0:
        return 0.5
    return (wins + 0.5 * draws) / total

def update_elo(r_a: float, r_b: float, s_a: float, cfg: EloConfig) -> Tuple[float, float]:
    e_a = expected_score(r_a, r_b, cfg)
    delta = cfg.k * (s_a - e_a)
    return r_a + delta, r_b - delta

@dataclass
class InternalEloState:
    ratings: Dict[str, float]
    cfg: EloConfig

def eval_and_update(
    state: InternalEloState,
    player_a_id: str,
    player_b_id: str,
    play_games_fn: Callable[[str, str, int], Tuple[int, int, int]],
    n_games: int,
) -> Dict[str, float]:
    ra = state.ratings.get(player_a_id, 1000.0)
    rb = state.ratings.get(player_b_id, 1000.0)

    wins, losses, draws = play_games_fn(player_a_id, player_b_id, n_games)
    s_a = score_from_counts(wins, losses, draws)

    ra2, rb2 = update_elo(ra, rb, s_a, state.cfg)
    state.ratings[player_a_id] = ra2
    state.ratings[player_b_id] = rb2

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": s_a,
        "elo_a_before": ra,
        "elo_b_before": rb,
        "elo_a_after": ra2,
        "elo_b_after": rb2,
        "elo_delta_a": ra2 - ra,
    }

# #REPRODUCIBILITY

# This section lists the external libraries required to run the notebook.
# 
# The implementation relies on standard numerical and deep learning tools for tensor computation and optimization, together with a chess engine library and the ReconChess framework to ensure correct handling of game rules and interaction between agents.
# 
# These dependencies provide the basic infrastructure for representing game states, running self-play matches, training neural networks, and evaluating the resulting agent.

# In[ ]:


def set_seeds(seed: int = 0) -> None:
    # This function makes runs more reproducible by fixing the random number generators (RNGs)
    # used for example for model initializationan and self-play variability.
    random.seed(seed) # Sets the seed for Python's built-in RNG.
    np.random.seed(seed) # Sets the seed for NumPy's RNG.
    torch.manual_seed(seed)
    # Sets the seed for PyTorch on CPU. This controls randomness in neural network weight
    # initialization and any CPU-based sampling operations, making training/debugging more repeatable.
    torch.cuda.manual_seed_all(seed)
    # Sets the seed for PyTorch on all available CUDA GPUs. This is important when training
    # or running inference on a GPU, so that GPU-side
    # randomness is also controlled as much as possible.


set_seeds(0)
# We call the function once at startup to fix a default seed. Using a fixed seed is useful for
# debugging (same behavior across runs), and reporting results (more consistent baselines).


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Automatically selects the computation device:
# - "cuda" means we will use the GPU (faster neural network inference/training when available),
# - otherwise we fall back to "cpu". This keeps the notebook portable across machines.

print("DEVICE:", DEVICE)
# Simple sanity check: prints whether we are actually running on GPU or CPU.
# This avoids accidentally running slow experiments on CPU when GPU was expected.


# ## ACTION ENCODING (20480 fixed policy head)
# 

# In order to use a fixed-size policy head, all possible chess moves are mapped to a discrete action space of fixed dimensionality.
# 
# Each move is encoded as an index in a predefined action set of size 20,480, covering all standard chess moves, including promotions. This encoding allows the policy network to produce a fixed-length output independent of the current position.
# 
# During play, only the subset of actions corresponding to legal moves provided by the game environment is considered, while the remaining entries are masked implicitly
# 

# In[ ]:


PROMO_TO_ID = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
# Maps a promotion piece type to a small integer ID.
# It includes None (no promotion) plus the 4 standard promotion options.
# This is needed because the same (from,to) move can represent different actions if promotion occurs.
ID_TO_PROMO = {v: k for k, v in PROMO_TO_ID.items()}
# Inverse mapping: converts the integer ID back to the promotion piece type.
# This is useful when we decode an action index back into an actual chess.Move.
POLICY_SIZE = 64 * 64 * 5  # 20480
# Total number of possible (from_square, to_square, promotion) combinations:
# - 64 possible origin squares
# - 64 possible destination squares
# - 5 promotion states (None + N/B/R/Q)
# This defines the fixed output size of the policy head.

def move_to_index(mv: chess.Move) -> int:
    # Converts a chess.Move into an integer index in [0, POLICY_SIZE).
    # This lets us store policies (π) as fixed-length vectors, and compare/learn them easily.
    f = mv.from_square
    # The origin square (0..63). python-chess uses a single integer for squares.

    t = mv.to_square
    # The destination square (0..63).
    pid = PROMO_TO_ID.get(mv.promotion, 0)
    # Promotion ID: if mv.promotion is None, we use 0. Otherwise map to 1..4.
    # This disambiguates different promotion actions.
    return (f * 64 + t) * 5 + pid

def index_to_move(idx: int) -> chess.Move:
    # Converts an integer action index back into a chess.Move.
    # This is needed to translate the policy / search choice into an actual move sent to ReconChess.
    pid = idx % 5
    # Extract promotion ID (remainder mod 5).
    x = idx // 5
    # Remove promotion part, leaving the packed from-to ID.
    f = x // 64
    # Recover the origin square (0..63).
    t = x % 64
    # Recover the destination square (0..63).
    promo = ID_TO_PROMO.get(pid, None)
    # Convert promo ID back to a piece type (or None).
    return chess.Move(from_square=f, to_square=t, promotion=promo)
    # Build a python-chess Move object.


# ## BELIEF TENSOR (7 channels) + SENSE UPDATE
# 

# Uncertainty about the opponent’s pieces is represented through a per-square belief tensor with seven channels, corresponding to the six standard chess piece types plus an explicit EMPTY channel.
# 
# For each board square, the belief tensor stores a probability distribution over these channels, normalized independently per square. This representation makes uncertainty explicit while remaining simple and easy to inspect.
# 
# Sensing actions update the belief deterministically within the sensed 3×3 region: observed squares are set to the corresponding piece type or to EMPTY, while beliefs outside the sensed area remain unchanged. This local update rule provides a lightweight mechanism to incorporate new information without maintaining a full probabilistic game history.

# Belief tensor: definitions

# In[ ]:


PIECE_TYPES_6 = ["P", "N", "B", "R", "Q", "K"]
# The six chess piece types (Pawn, Knight, Bishop, Rook, Queen, King).
# We represent uncertainty only about the opponent’s pieces, so we track probabilities over these types.

EMPTY = "EMPTY"
# Extra channel used to represent the probability that a square contains no opponent piece.
# In RBC we do not see the opponent, so "empty vs occupied" is itself uncertain.

CHANNELS_7 = PIECE_TYPES_6 + [EMPTY]
# Full set of belief channels: 6 piece types + EMPTY.
# For each square, the belief distribution over CHANNELS_7 should sum to 1.

C_BELIEF = 7
# Number of belief channels.

CH2I = {ch: i for i, ch in enumerate(CHANNELS_7)}
# Channel-to-index mapping, so we can access the right slice of the belief tensor by name.
# Example: CH2I["Q"] gives the channel index for queens.

START_COUNTS = {"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1}
# Initial piece inventory for a standard chess starting position.
# We use this as a lightweight constraint during determinization:
# the opponent cannot have more pieces of a type than physically possible.


# Normalization over channels (to make it a true distribution)

# In[ ]:


def normalize_over_channels(B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # B has shape (C, 8, 8), where C is the number of channels (7 here).
    # For each board square (r, f), we want sum over channels to be 1:
    #   sum_c B[c, r, f] = 1
    # This is essential because we interpret B[:, r, f] as a probability distribution.

    s = B.sum(dim=0, keepdim=True).clamp_min(eps)
    # s has shape (1, 8, 8) and contains, for each square, the sum over channels.
    # clamp_min(eps) prevents division by zero (numerical safety).

    return B / s
    # Divides each channel value by the per-square sum, making each square a valid distribution.


# In[ ]:


def piece_symbol_upper(pc: chess.Piece) -> str:
    # python-chess represents pieces with symbols: 'p', 'n', ... (lowercase for black).
    # We convert to uppercase ('P','N',...) so that the belief channels use a consistent convention.
    return pc.symbol().upper()


# Belief's initialization from the standard board position

# In[ ]:


def init_belief_from_initial(my_color: chess.Color) -> torch.Tensor:
    """
    RBC starts from the standard chess initial position.
    We initialize the opponent belief with a strong prior: probability 1 on the opponent’s
    initial piece placement, and EMPTY elsewhere.

    This is a reasonable baseline because the starting configuration is known in RBC;
    only after the game starts the opponent becomes hidden.
    """
    B = torch.zeros((C_BELIEF, 8, 8), dtype=torch.float32)
    # Create an empty belief tensor of shape (7, 8, 8).

    B[CH2I[EMPTY]].fill_(1.0)
    # Default assumption: every square is empty of opponent pieces (probability 1).
    # We will overwrite the squares where the opponent initially has pieces.

    board = chess.Board()
    # Creates a standard chess board in the initial position.
    # This is used only to read the known initial placement.

    opp_color = not my_color
    # Opponent color is the opposite of our color.

    for sq, pc in board.piece_map().items():
        # Iterate over all pieces on the initial board.

        if pc.color != opp_color:
            # Skip our own pieces: the belief tensor models only opponent occupancy.
            continue

        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        # Convert a square index (0..63) into board coordinates (rank/file) in [0..7].

        sym = piece_symbol_upper(pc)
        # Convert the piece to a channel symbol: 'P','N','B','R','Q','K'.

        B[:, r, f] = 0.0
        # Clear the distribution at that square: we will set it to a one-hot belief.

        B[CH2I[sym], r, f] = 1.0
        # Set probability 1 for the correct opponent piece type at its initial location.

    return normalize_over_channels(B)
    # Final safety: ensures per-square distributions sum to 1.


# Updates the opponent belief tensor after a sensing action.

# In[ ]:


def apply_sense_to_belief(
    B: torch.Tensor,
    sense_result: List[Tuple[Square, Optional[chess.Piece]]],
    my_color: chess.Color
) -> torch.Tensor:
    #The sensing action reveals the exact contents of a 3x3 window centered on sense_square.
    #For squares inside this window, the belief becomes deterministic.
    #For all other squares, the belief remains unchanged.
    """Update opponent belief using ReconChess sense results (copy-based, not in-place)."""
    B2 = B.clone() # Copy the belief tensor to avoid side effects and make debugging/search safer.
    for sq, pc in sense_result: # Iterate over each sensed square and its observed content.
        r = chess.square_rank(sq) # Convert square index to rank (0..7) for tensor indexing.
        f = chess.square_file(sq) # Convert square index to file (0..7) for tensor indexing.
        B2[:, r, f] = 0.0 # Clear previous uncertainty at this square before setting a deterministic observation.
        if pc is None or pc.color == my_color: # Empty square or our own piece => no opponent piece present here.
            B2[CH2I[EMPTY], r, f] = 1.0 # Set EMPTY with probability 1 (opponent occupancy is certainly empty).
        else: # Otherwise the observed piece must belong to the opponent.
            sym = piece_symbol_upper(pc)  # Map the observed piece to a channel label (P/N/B/R/Q/K).
            B2[CH2I[sym], r, f] = 1.0 # Set a one-hot belief for that opponent piece type at this square.
    return normalize_over_channels(B2) # Ensure per-square probabilities sum to 1, then return the updated belief.


# ## SENSE SELECTION: entropy-max 3×3 center
# 

# At each turn, the agent selects a sensing action by evaluating the uncertainty of the opponent’s belief distribution.
# 
# For each allowed sensing square, the total entropy over the corresponding 3×3 region is computed, and the square that maximizes this value is selected. This heuristic prioritizes sensing actions that are expected to provide the largest reduction in uncertainty.
# 
# The approach is purely information-driven and independent of the immediate move selection, making it simple, efficient, and consistent with the belief representation.

# In[ ]:


SENSE_OFFSETS = [(dr, df) for dr in (-1,0,1) for df in (-1,0,1)]
# Relative offsets for the 3x3 sensing window (center plus its 8 neighbors)

def square_entropy(p: torch.Tensor, eps: float = 1e-8) -> float:
  # Compute Shannon entropy of a probability vector p (uncertainty measure).
    p = p.clamp_min(eps)
    # Avoid log(0) for numerical stability (sensing often creates exact zeros/ones).
    return float(-(p * p.log()).sum().item())
    # Shannon entropy: higher means more uncertainty about what occupies this square.


def choose_sense_square_entropy(B: torch.Tensor, sense_actions: List[Square]) -> Square:
  # Select a sensing square by maximizing expected information gain (proxy: entropy).
    """Choose among *allowed* sense_actions provided by ReconChess."""
    best_sq = sense_actions[0]
    # Initialize with a valid default action to ensure we always return a legal sense square.
    best_e = -1e18
    # Track the best (maximum) entropy score; start from a very low value.

    for sq in sense_actions:
        # Evaluate each legal sensing action and pick the one with the highest total uncertainty.

        r0 = chess.square_rank(sq)
        # Convert the candidate sensing square to rank coordinate (0..7).
        f0 = chess.square_file(sq)
        # Convert the candidate sensing square to file coordinate (0..7).

        tot = 0.0
        # Accumulate entropy over the 3x3 window: higher total => more uncertainty to resolve.
        for dr, df in SENSE_OFFSETS:
          # Iterate over the 3x3 neighborhood that would be revealed by sensing at sq.
            r = r0 + dr; f = f0 + df
            # Compute the coordinates of each square inside the sensing window.
            if 0 <= r < 8 and 0 <= f < 8:
               # Only consider squares that lie within the board boundaries.
                tot += square_entropy(B[:, r, f])
                # Add uncertainty of this square; sensing is more valuable where belief is uncertain.
        if tot > best_e:
          # Keep the sensing action that maximizes total entropy (information gain proxy).
            best_e = tot
            # Update best entropy score.
            best_sq = sq
            # Update best sensing square.

    return best_sq
    # Return the legal sensing action expected to reduce uncertainty the most.


# ## GREEDY DETERMINIZATION FROM BELIEF + remaining opponent inventory
# 

# To enable fast planning with standard chess move generation, the opponent’s hidden position is approximated by constructing a single fully specified “determinized” board state from the belief tensor.
# 
# For each opponent piece type, the algorithm places the remaining pieces on the highest-probability squares according to the belief distribution, while respecting already occupied squares (including all known own pieces). A simple opponent inventory is maintained to ensure that the determinized position contains a consistent number of pieces of each type.
# 
# The resulting determinized board is used only as a hypothesis for search and evaluation; it provides a concrete state on which legal moves can be checked and simulated efficiently.

# Determinize the hidden opponent position from belief.

# In[ ]:


def greedy_determinize_opponent(
    # Build a single deterministic opponent placement consistent with belief and piece counts.
    B: torch.Tensor, # Belief tensor (channels x 8 x 8) encoding opponent occupancy probabilities.
    own_board: chess.Board,  # Board containing our known pieces (fully observable in RBC).
    my_color: chess.Color,  # Our color; opponent color is the opposite.
    opp_counts: Dict[str, int], # Remaining opponent inventory per piece type (P/N/B/R/Q/K).
) -> chess.Board: # Returns a fully specified deterministic board used for planning/search.

    det = chess.Board(None) # Create an empty python-chess board (no default starting position).
    det.clear() # Ensure the board starts completely empty before placing pieces.

    # Place our own pieces deterministically (we always know our pieces in RBC).
    for sq, pc in own_board.piece_map().items(): # Iterate over our known piece locations.
        det.set_piece_at(sq, pc) # Copy our piece into the determinized board.

    occupied = set(own_board.piece_map().keys()) # Squares already occupied by our pieces.
    opp_color = not my_color # Opponent color (WHITE <-> BLACK).

    for sym in PIECE_TYPES_6: # Place opponent pieces type-by-type (P,N,B,R,Q,K).
        k = int(opp_counts.get(sym, 0)) # How many opponent pieces of this type are still possible.
        if k <= 0:
            continue # Skip piece types that are no longer present in the opponent inventory.

        probs = [] # Collect candidate squares with their belief probability for this piece type.
        ch = CH2I[sym] # Channel index corresponding to this piece type in the belief tensor.
        for sq in chess.SQUARES: # Consider every board square as a potential placement.
            if sq in occupied:
                continue # Cannot place opponent pieces on already occupied squares.
            r = chess.square_rank(sq) # Rank coordinate for belief indexing.
            f = chess.square_file(sq) # File coordinate for belief indexing.
            probs.append((float(B[ch, r, f].item()), sq)) # Store (belief probability, square) pairs.
        probs.sort(reverse=True, key=lambda x: x[0]) # Sort candidate squares by descending probability.

        placed = 0 # Counter for how many pieces of this type we have placed so far.
        for _, sq in probs: # Greedily iterate squares from highest to lowest probability.
            if sq in occupied: # Safety check: skip squares that became occupied meanwhile.
                continue
            # Create a python-chess piece symbol with the correct opponent color.

            psym = sym.lower() if opp_color == chess.BLACK else sym # python-chess uses lowercase for black pieces and uppercase for white pieces.
            det.set_piece_at(sq, chess.Piece.from_symbol(psym))  # Place the determinized opponent piece on the board at this square.
            occupied.add(sq) # Mark square as occupied so no other piece is placed here.
            placed += 1 # One more piece of this type has been placed.
            if placed >= k:
                break # Stop once we placed the required number of pieces of this type.

    det.turn = own_board.turn # Keep the correct side-to-move for planning/search.
    det.castling_rights = own_board.castling_rights # Preserve castling rights metadata for legality.
    return det # Return the determinized full-information board used as a hypothesis for search.


# In[ ]:


def apply_taken_move_to_own_board(own_board: chess.Board, mv: chess.Move, my_color: chess.Color) -> None:
  # Apply the resolved move (taken_move) to our internal board that tracks only OUR pieces.
  # own_board is intentionally partial: it stores only our pieces (opponent pieces are unknown/hidden).
  # We avoid python-chess board.push() because it assumes a full legal position with both sides' pieces.
  # Manual update covers the minimum needed mechanics for our piece tracking (move, promotion, castling, metadata).



    pc = own_board.piece_at(mv.from_square) # Get our piece located at the move origin square.
    if pc is None or pc.color != my_color: # If we don't find our own piece at the origin, we skip the update (safer than guessing).
        return # Safety: avoid corrupting our internal state when tracking is inconsistent.


    # Remove the moving piece from its origin square.
    own_board.remove_piece_at(mv.from_square)

    # Handle pawn promotion if the resolved move includes a promotion piece type
    if mv.promotion is not None and pc.piece_type == chess.PAWN: # Promotions happen when a pawn reaches the last rank and is promoted to another piece.
        pc = chess.Piece(mv.promotion, my_color) # Replace pawn with the promoted piece type.

    # Place the (possibly promoted) piece onto the destination square.
    own_board.set_piece_at(mv.to_square, pc)  # Update our internal board with the moved piece.

    # Handle castling: detect it when the king moves exactly two files and move the rook accordingly
    if pc.piece_type == chess.KING: # Only king moves can represent castling.
        f_from = chess.square_file(mv.from_square) # Origin file of the king move.
        f_to = chess.square_file(mv.to_square)  # Destination file of the king move.
        r_rank = chess.square_rank(mv.from_square) # Castling rook stays on the same rank as the king.
        if abs(f_to - f_from) == 2: # King moved two files: this is castling (kingside or queenside).
            if f_to > f_from:  # Kingside castling: rook moves from file 7 to file 5.
                rook_from = chess.square(7, r_rank) # Rook starts at the corner (h-file).
                rook_to = chess.square(5, r_rank) # Rook ends next to the king (f-file)
            else: # Queenside castling: rook moves from file 0 to file 3.
                rook_from = chess.square(0, r_rank) # Rook starts at the corner (a-file).
                rook_to = chess.square(3, r_rank) # Rook ends next to the king (d-file).

            rook = own_board.piece_at(rook_from) # Fetch the rook we expect to move during castling.
            if rook is not None and rook.color == my_color and rook.piece_type == chess.ROOK: # Safety check: only move the rook if it exists and is our rook.
                own_board.remove_piece_at(rook_from) # Clear rook origin square.
                own_board.set_piece_at(rook_to, rook) # Place rook on its castling destination square.

    # Update turn/fullmove (approx, good enough for metadata planes)
    moving_color = own_board.turn  # color to move BEFORE applying the move

    own_board.turn = not own_board.turn
    if moving_color == chess.BLACK:
        own_board.fullmove_number += 1


# ## ENCODER (own pieces + belief + small metadata) → 15×8×8
# 

# The neural network input is a stack of 2D feature planes with fixed spatial resolution (8×8), producing a tensor of shape 15×8×8.
# 
# The encoding includes: (i) six binary planes for the agent’s own pieces (one per piece type), (ii) the seven-channel opponent belief tensor, and (iii) a small set of global metadata planes (side to move and a normalized move counter).
# 
# This representation keeps the input compact while preserving the spatial structure of the board, allowing convolutional layers to exploit local patterns and piece configurations.

# In[ ]:


PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
# Fixed ordering of piece types to map each piece to a consistent channel index (used in state encoding).

def board_to_own_planes(board: chess.Board, my_color: chess.Color) -> torch.Tensor:
    # Encode our visible pieces as 6 binary 8x8 planes (one plane per piece type) for CNN input.
    X = torch.zeros((6, 8, 8), dtype=torch.float32)
    # X[i, r, f] = 1 indicates that our piece of type i occupies square (r, f).

    for sq, pc in board.piece_map().items():
        # Iterate over pieces on the board representation.
        if pc.color != my_color:
            continue
        # Only encode our own pieces; opponent uncertainty is handled by the belief tensor B.

        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        # Convert square index to (rank, file) coordinates for indexing an 8x8 grid.

        i = PIECE_ORDER.index(pc.piece_type)
        # Map the piece type to a plane index (pawn=0, knight=1, ..., king=5).

        X[i, r, f] = 1.0
        # Set a one-hot marker for the presence of this piece type at this location.

    return X
    # Return the 6-channel encoding of our fully observable pieces.

def metadata_planes(board: chess.Board, my_color: chess.Color) -> torch.Tensor:
    # Create constant planes that provide global context (turn and game progress) to the CNN.
    turn_plane = torch.full((1,8,8), 1.0 if board.turn == my_color else 0.0, dtype=torch.float32)
    # Turn plane: 1 everywhere if it is our turn, else 0 everywhere.

    ply = min(board.fullmove_number * 2, 200) / 200.0
    # Approximate ply count (2 * fullmove_number), clipped to 200 and normalized to [0,1].

    ply_plane = torch.full((1,8,8), float(ply), dtype=torch.float32)
    # Game-progress plane: constant across the board, giving the network a timing signal.

    return torch.cat([turn_plane, ply_plane], dim=0)
    # Stack metadata planes into shape (2, 8, 8).

def encode_state(own_board: chess.Board, my_color: chess.Color, B: torch.Tensor) -> torch.Tensor:
    # Build the full NN input by concatenating: own piece planes (6), belief planes (7), metadata planes (2).
    own = board_to_own_planes(own_board, my_color)
    # Encode our visible pieces as 6 channels.

    meta = metadata_planes(own_board, my_color)
    # Encode global context as 2 channels.

    return torch.cat([own, B, meta], dim=0)  # (15,8,8)
    # Final encoding: 15 channels total, ready to be fed into the policy/value CNN.


# ## SMALL POLICY/VALUE NET
# 

# The agent uses a lightweight convolutional neural network with a shared trunk and two output heads: a policy head and a value head.
# 
# The policy head produces logits over the fixed 20,480-action encoding, which are later restricted to the legal moves available in the current position. The value head outputs a single scalar estimating the expected game outcome from the current player’s perspective.
# 
# The network is intentionally small to keep self-play and training fast while still capturing the spatial structure of the board representation.

# In[ ]:


class FastPolicyValueNet(nn.Module):
    # Lightweight AlphaZero-style network: shared CNN trunk + policy head + value head.
    def __init__(self, in_ch: int = 15, trunk: int = 64):
        # in_ch = number of input planes (15); trunk = number of channels in the shared CNN trunk.
        super().__init__()  # Initialize nn.Module so PyTorch can track parameters and submodules.

        self.trunk = nn.Sequential(
            # Shared convolutional feature extractor (used by both policy and value heads).
            nn.Conv2d(in_ch, trunk, 3, padding=1),
            # 3x3 conv keeps spatial size (8x8) and maps input planes to trunk feature maps.
            nn.ReLU(),
            # Non-linearity to allow learning rich representations.
            nn.Conv2d(trunk, trunk, 3, padding=1),
            # Second 3x3 conv refines features while keeping the same channel width.
            nn.ReLU(),
            # Another non-linearity; keeps the trunk lightweight but expressive.
        )

        self.pol = nn.Sequential(nn.Conv2d(trunk, 32, 1), nn.ReLU())
        # Policy head: 1x1 conv compresses trunk features into a smaller policy-specific representation.
        self.pol_fc = nn.Linear(32*8*8, POLICY_SIZE)
        # Final policy layer: outputs logits over the fixed action space (POLICY_SIZE = 20480).

        self.val = nn.Sequential(nn.Conv2d(trunk, 16, 1), nn.ReLU())
        # Value head: 1x1 conv builds a compact feature representation for scalar value prediction.
        self.val_fc1 = nn.Linear(16*8*8, 64)
        # Value MLP part: map flattened value features into a small hidden vector.
        self.val_fc2 = nn.Linear(64, 1)
        # Output a single scalar value (later squashed to [-1, 1] with tanh).

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward pass: returns (policy_logits, value) for a batch of encoded states.
        h = self.trunk(x)
        # Shared feature maps extracted from the input state.

        logits = self.pol_fc(self.pol(h).flatten(1))
        # Policy: head features -> flatten -> linear layer -> logits over all actions (before masking legal moves).

        v = torch.tanh(self.val_fc2(F.relu(self.val_fc1(self.val(h).flatten(1))))).squeeze(-1)
        # Value: predict a scalar in [-1, 1] (tanh), matching win/loss targets used in self-play training.

        return logits, v
        # Return policy logits (for action priors) and value estimate (for leaf evaluation and training).


# ## ROOT-only PUCT (search on determinization, choose among provided legal actions)
# 

# Move selection is performed using a lightweight, root-only PUCT search guided by the network’s policy priors.
# 
# The search is run on the determinized board hypothesis and considers only the move actions provided by the ReconChess environment for the current turn. Each simulation selects the move that maximizes a PUCT score combining an exploitation term (estimated value) and an exploration term weighted by the network prior.
# 
# The final move is chosen from the resulting visit counts, producing a search-improved policy target that is also reused during training.

# In[ ]:


@torch.no_grad() # Disable gradient tracking for faster and memory-efficient inference during search.
def nn_priors_and_value(model: nn.Module, x: torch.Tensor, legal_moves: List[chess.Move], device: str) -> Tuple[Dict[int,float], float]:
  # Run the policy/value network once to obtain priors over legal moves and a value estimate for the state.
    model.eval() # Switch to evaluation mode for stable inference (no dropout, consistent behavior).
    xb = x.unsqueeze(0).to(device) # Add batch dimension (1, C, 8, 8) and move input to the chosen device (CPU/GPU).
    logits, v = model(xb) # Forward pass: logits over the full action space + scalar value estimate for the state.
    logits = logits[0].cpu() # Remove batch dimension and bring logits to CPU for indexing legal actions.
    v = float(v.item()) # Convert the value tensor into a plain Python float.

    idxs = [move_to_index(m) for m in legal_moves] # Map each legal chess.Move to its fixed action index in the policy vector.
    l = logits[idxs] # Extract logits only for legal actions (masking the full action space down to legal moves).
    p = torch.softmax(l, dim=0).numpy() # Convert legal-move logits into a probability distribution (priors) using softmax.
    pri = {idxs[i]: float(p[i]) for i in range(len(idxs))} # Build a dict mapping action index -> prior probability for quick access inside PUCT.
    return pri, v # Return priors over legal actions and the value estimate for the current state.

@torch.no_grad()
def value_after_move(
    model: nn.Module,
    own_board: chess.Board,
    my_color: chess.Color,
    B: torch.Tensor,
    mv: chess.Move,
    device: str,
) -> float:
    """One-step value lookahead: V(s after mv), belief unchanged."""
    own2 = own_board.copy()
    apply_taken_move_to_own_board(own2, mv, my_color)
    x2 = encode_state(own2, my_color, B).unsqueeze(0).to(device)
    _, v2 = model(x2)
    return float(v2.item())

@torch.no_grad()
def puct_root(
    model: nn.Module,
    x_root: torch.Tensor,
    own_board: chess.Board,
    my_color: chess.Color,
    B: torch.Tensor,
    move_actions: List[chess.Move],
    sims: int = 80,
    c_puct: float = 1.5,
    device: str = "cpu",
) -> Dict[int, int]:

    if not move_actions:
        return {}

    # Priors over legal actions
    priors, _ = nn_priors_and_value(model, x_root, move_actions, device=device)

    # Precompute one-step value for each legal move
    # Precompute one-step value only for Top-K moves by prior (speed)
    K = 8  # prova 8, 12, 16
    pairs = [(priors.get(move_to_index(m), 0.0), m) for m in move_actions]
    pairs.sort(reverse=True, key=lambda x: x[0])
    top_moves = [m for _, m in pairs[:min(K, len(pairs))]]

    v_child: Dict[int, float] = {move_to_index(m): 0.0 for m in move_actions}
    for m in top_moves:
        a = move_to_index(m)
        v_child[a] = value_after_move(model, own_board, my_color, B, m, device=device)

    N = 0
    N_a: Dict[int, int] = {move_to_index(m): 0 for m in move_actions}
    W_a: Dict[int, float] = {move_to_index(m): 0.0 for m in move_actions}

    def Q(a: int) -> float:
        n = N_a[a]
        return 0.0 if n == 0 else W_a[a] / n

    for _ in range(sims):
        N += 1
        best_a = None
        best_s = -1e18

        for m in move_actions:
            a = move_to_index(m)
            u = c_puct * priors.get(a, 0.0) * math.sqrt(N) / (1 + N_a[a])
            s = Q(a) + u
            if s > best_s:
                best_s = s
                best_a = a

        v_leaf = v_child.get(best_a, 0.0)

        N_a[best_a] += 1
        W_a[best_a] += float(v_leaf)

    return N_a

def visits_to_policy_target(N_a: Dict[int,int]) -> np.ndarray:
  # Convert visit counts N(a) into a full-size policy target vector π over the fixed action space.
    P = np.zeros((POLICY_SIZE,), dtype=np.float32) # Initialize a dense policy vector; only visited/legal action indices will receive non-zero mass.
    if not N_a:
        return P # No visits available; return an all-zero target (edge case).
    total = sum(N_a.values()) # Total number of visits across actions, used to normalize counts into probabilities.
    for a, n in N_a.items(): # Fill the policy vector at action index a with the normalized visit frequency.
        P[a] = n / max(1, total) # Normalize visit counts into a probability distribution π(a).
    return P # Return the AlphaZero-style policy target derived from search visits.


# ## THE **RECONCHESS PLAYER** (FAST, rule-compliant)
# 

# The agent is implemented as a ReconChess Player, fully compliant with the game’s interface and rules.
# 
# All game interactions—including sensing, move selection, belief updates, and board state tracking—are handled through the standard ReconChess callbacks. This ensures that the agent can play against other bots without relying on privileged information or modified game mechanics. The focus is on robustness and correct interaction rather than maximal performance.
# 

# In[ ]:


@dataclass
class FastBotConfig:
    # Configuration container for the FAST AlphaZero-like RBC bot.
    sims: int = 80      # Number of root PUCT simulations per move (higher = stronger but slower).
    c_puct: float = 1.5 # Exploration constant controlling the PUCT exploration bonus.


class RBCFastAZPlayer(Player):
    # ReconChess-compatible player that combines belief modeling + greedy determinization + root PUCT.
    def __init__(self, model: nn.Module, cfg: FastBotConfig, seed: int = 0):
        self.model = model  # Policy/value network used for priors and value estimation.
        self.cfg = cfg      # Search hyperparameters (sims, c_puct, etc.).
        self.rng = np.random.default_rng(seed)  # Reproducible RNG for fallback random decisions.

        # Will be set in handle_game_start
        self.color: Optional[chess.Color] = None        # Our color (python-chess boolean convention).
        self.own_board: Optional[chess.Board] = None    # Internal board tracking ONLY our pieces.
        self.B: Optional[torch.Tensor] = None           # Belief tensor for opponent occupancy (7,8,8).
        self.opp_counts: Optional[Dict[str,int]] = None # Remaining opponent piece inventory by type.

        # Optional training buffers (can be disabled)
        self.store_training = False   # If True, store (X, π, z) tuples for offline training.
        self.X: List[np.ndarray] = [] # Stored encoded states.
        self.P: List[np.ndarray] = [] # Stored policy targets derived from search visit counts.
        self.Z: List[float] = []      # Stored value targets (final game outcome).

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        # Called once at the beginning of the game.
        self.color = bool(color)  # Convert ReconChess Color to python-chess color boolean.

        # In ReconChess, we always know our own pieces; the provided board is standard initial chess setup.
        self.own_board = chess.Board(None)
        self.own_board.clear()  # Keep an internal board containing ONLY our pieces.

        for sq, pc in board.piece_map().items():
            if pc.color == self.color:
                self.own_board.set_piece_at(sq, pc)  # Copy only our pieces to the internal board.

        self.own_board.turn = board.turn                  # Preserve side-to-move metadata.
        self.own_board.castling_rights = board.castling_rights  # Preserve castling rights metadata.

        self.B = init_belief_from_initial(self.color)  # Strong prior: opponent starts from standard setup.
        self.opp_counts = dict(START_COUNTS)           # Initialize opponent inventory to standard counts.

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # After the opponent move, we only learn whether they captured one of our pieces and where.
        if captured_my_piece and capture_square is not None:
            if self.own_board is not None:
                self.own_board.remove_piece_at(capture_square)  # Remove our captured piece from internal tracking.

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Square:
        # Choose a sensing action among those allowed by ReconChess.
        assert self.B is not None
        return choose_sense_square_entropy(self.B, sense_actions)  # Greedy: sense where belief entropy is highest.

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # Update belief tensor with the observed contents of the 3x3 sensing window.
        assert self.B is not None and self.color is not None
        self.B = apply_sense_to_belief(self.B, sense_result, self.color)  # Copy-based belief update.

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # Choose a move action among those allowed by ReconChess.
        assert self.own_board is not None and self.color is not None and self.B is not None and self.opp_counts is not None

        if not move_actions:
            return None  # No legal move actions available.

        x_root = encode_state(self.own_board, self.color, self.B)  # Encode (own pieces + belief + metadata).

        N_a = puct_root(
            model=self.model,
            x_root=x_root,
            own_board=self.own_board,
            my_color=self.color,
            B=self.B,
            move_actions=move_actions,
            sims=self.cfg.sims,
            c_puct=self.cfg.c_puct,
            device=DEVICE,
        )
        # Run root-only PUCT guided by NN priors; returns visit counts over action indices.

        if self.store_training:
            # Store training targets (X, π). Value targets z are filled in handle_game_end.
            self.X.append(x_root.detach().cpu().numpy().astype(np.float32))
            self.P.append(visits_to_policy_target(N_a))

        # Choose the move with the highest visit count (AlphaZero style); fallback to random legal.
        if N_a:
            best_a = max(N_a.items(), key=lambda kv: kv[1])[0]  # Action index with highest visits.
            mv = index_to_move(best_a)                          # Decode back into a chess.Move.
            if mv in move_actions:
                return mv  # Return chosen move if still legal/allowed by ReconChess.

        return move_actions[int(self.rng.integers(0, len(move_actions)))]  # Random fallback (always legal).

    def handle_move_result(
        self,
        requested_move: Optional[chess.Move],
        taken_move: Optional[chess.Move],
        captured_opponent_piece: bool,
        capture_square: Optional[Square],
    ):
        # Called after our move: ReconChess provides both the requested move and the resolved taken move.
        if self.own_board is None or self.color is None:
            return  # Safety: if not initialized, do nothing.

        if taken_move is not None:
            # Update our own-piece board using the move that was actually executed (not just requested).
            apply_taken_move_to_own_board(self.own_board, taken_move, self.color)

        # If we captured an opponent piece, we update the opponent inventory in a FAST approximate way.
        if captured_opponent_piece and capture_square is not None and self.opp_counts is not None and self.B is not None:
            r = chess.square_rank(capture_square)
            f = chess.square_file(capture_square)
            # Convert capture square into coordinates so we can consult belief probabilities at that location.

            probs = self.B[:, r, f].detach().cpu()
            # Read the belief distribution over piece types at the capture square.

            best_sym = None
            best_p = -1.0

            # 1) Pick the most likely non-EMPTY piece type that is still available in opponent inventory.
            for sym in PIECE_TYPES_6:
                p = float(probs[CH2I[sym]].item())
                if p > best_p and self.opp_counts.get(sym, 0) > 0:
                    best_p = p
                    best_sym = sym

            # 2) If belief is uninformative, fall back to a simple heuristic (prefer pawns first).
            if best_sym is None:
                if self.opp_counts.get("P", 0) > 0:
                    best_sym = "P"
                else:
                    for sym in ["N", "B", "R", "Q", "K"]:
                        if self.opp_counts.get(sym, 0) > 0:
                            best_sym = sym
                            break

            # 3) Decrement the selected type in the remaining-inventory estimate.
            if best_sym is not None:
                self.opp_counts[best_sym] = max(0, self.opp_counts.get(best_sym, 0) - 1)

    def handle_game_end(self, winner_color: Optional[Color], reason: str, game_history: Any):
        # Called at the end of the game. If training buffers are enabled, fill value targets z for each stored state.
        if not self.store_training or self.color is None:
            return  # Training disabled or not initialized.

        if winner_color is None:
            z = 0.0  # Draw or unknown result.
        else:
            z = 1.0 if bool(winner_color) == self.color else -1.0  # Win = +1, loss = -1 from our perspective.

        self.Z = [z] * len(self.X)
        # Assign the final outcome to all states from this episode (AlphaZero-style episodic value target).


# ## LOCAL MATCH HARNESS (ReconChess) — smoke test
# 

# A local match harness is used to run short games against a baseline opponent, serving as an end-to-end smoke test for rule compliance and framework integration.

# In[ ]:


def try_discover_play_local_game():
    # ReconChess utilities moved across versions; this helper probes common locations for play_local_game.
    try:
        from reconchess.utilities import play_local_game
        return play_local_game
    except Exception:
        pass  # If this import path is unavailable, try another known location.

    try:
        from reconchess.scripts.rc_play_game import play_local_game
        return play_local_game
    except Exception:
        pass  # Fallback for older/newer ReconChess layouts.

    try:
        from reconchess.scripts.rc_bot_match import play_local_game
        return play_local_game
    except Exception:
        pass

    raise ImportError(
        "Could not find play_local_game in reconchess. Install a version that provides local runner utilities."
    )
    # Fail fast with a clear error message if no local runner is available.


def smoke_test_vs_random(n_games: int = 5, seed: int = 0):
    # Quick end-to-end sanity check: play a few local games against RandomBot and count outcomes.
    play_local_game = try_discover_play_local_game()
    # Obtain the local game runner in a version-robust way.

    model = FastPolicyValueNet(in_ch=15, trunk=64).to(DEVICE)
    # Create the policy/value network and move it to the configured device (CPU/GPU).

    cfg = FastBotConfig(sims=40, c_puct=1.5)
    # Use fewer PUCT simulations for a fast smoke test (not for peak playing strength).

    wins = 0
    losses = 0
    draws = 0
    # Track outcomes to verify that games complete and the integration works.

    for g in range(n_games):
        # Play multiple games to test stability across different seeds.
        bot = RBCFastAZPlayer(model=model, cfg=cfg, seed=seed + g)
        # Instantiate our bot; vary RNG seed per game for reproducible but diverse behavior.

        opp = reconchess.bots.random_bot.RandomBot()
        # Baseline opponent for sanity checking.

        winner_color, reason, history = play_local_game(bot, opp)
        # Run one local game and retrieve winner, termination reason, and full game history.

        if winner_color is None:
            draws += 1
            # No winner reported (draw or runner-specific termination).
        elif bool(winner_color) == chess.WHITE:  # our bot is usually Player 1 (White) in many local runners
            wins += 1
            # Under the common assumption that our bot plays White, a White win counts as our win.
        else:
            losses += 1
            # Otherwise, count as a loss under the same assumption.

        print(f"[game {g}] winner={winner_color} reason={reason}")
        # Per-game debug printout.

    print({"wins": wins, "losses": losses, "draws": draws})
    # Print a final summary of results.


# ### Run smoke test (uncomment)
# 

# In[ ]:


# smoke_test_vs_random(n_games=3, seed=0)


# ## RUN ALL CHECKS (fast sanity gate)
# These checks are meant to fail fast if something is inconsistent. If they pass, the agent is generally safe to run in local matches and self-play
# 

# In[ ]:


def run_all_checks():
    # Run a small set of sanity checks to validate core invariants before self-play/training.

    # 1) Belief normalization check: for every square, probabilities across channels should sum to 1.
    model = FastPolicyValueNet(in_ch=15, trunk=64).to(DEVICE)
    # Create the policy/value network and move it to the selected device (CPU/GPU).

    bot = RBCFastAZPlayer(model=model, cfg=FastBotConfig(sims=10, c_puct=1.5), seed=0)
    # Instantiate the bot with a small number of PUCT simulations for a fast unit test.

    start_board = chess.Board()
    # Standard full-information chess starting position (used here only as a convenient initializer).

    bot.handle_game_start(color=chess.WHITE, board=start_board, opponent_name="opp")
    # Simulate ReconChess game start so the bot initializes own_board, belief tensor, and opponent inventory.

    assert bot.B is not None
    # Ensure belief tensor was initialized.

    s = bot.B.sum(dim=0)
    # Sum over channels -> should be an 8x8 grid of ones if belief is normalized per square.

    assert float((s - 1.0).abs().max().item()) < 1e-4, "Belief not normalized per square"
    # Check maximum deviation from 1.0 across all squares (numerical tolerance).

    # 2) choose_sense returns an allowed action (sense square must belong to sense_actions).
    sense_actions = list(range(64))
    # For testing we allow sensing on any square (ReconChess may restrict this in actual games).

    mv_actions = list(start_board.legal_moves)
    # Provide a non-empty move list to match the choose_sense signature (bot may ignore it).

    sq = bot.choose_sense(sense_actions, mv_actions, seconds_left=100.0)
    # Ask the bot to select a sensing action.

    assert sq in sense_actions, "choose_sense returned illegal square"
    # Verify returned sense square is legal according to the provided action set.

    # 3) choose_move returns an allowed move (must be in move_actions or None if no moves exist).
    mv = bot.choose_move(mv_actions, seconds_left=100.0)
    # Ask the bot to select a move among the provided legal moves.

    assert (mv is None) or (mv in mv_actions), "choose_move returned move not in move_actions"
    # Verify the move returned by the bot belongs to the allowed action list.

    print("Core invariants: OK")
    # If we reached here, basic invariants are satisfied.

    # 4) Smoke match vs RandomBot (optional): tests end-to-end integration with a local game runner.
    try:
# smoke_test_vs_random(n_games=2, seed=0)
        # Run a couple of local games vs a RandomBot to ensure the full pipeline executes without crashing.
        print("Smoke games: OK")
    except Exception as e:
        # In some environments (e.g., Colab), local runner utilities may be missing; do not fail hard.
        print("Smoke games: SKIPPED/FAILED (environment issue):", e)


if __name__ == "__main__":
    # run_all_checks()
    pass
# Execute all checks immediately when the cell runs.


# # SELF-PLAY + TRAINING (FAST)
# 
# This section makes the bot trainable with minimal extra code:
# 
# 1) ReplayBuffer in RAM (FAST)
# 2) Training step: KL(policy) + MSE(value)
# 3) Self-play game generator (ReconChess local runner)
# 4) Iterative loop: self-play → train → eval → checkpoint
# 

# ## ReplayBuffer (RAM) + Dataset
# 

# Self-play generates training samples over time, so the implementation stores them in a replay buffer kept in RAM.
# 
# The buffer collects tuples (X,π,z), where X is the encoded state, π is the search-improved policy target, and z is the final game outcome from the player’s perspective. A maximum capacity is enforced by discarding the oldest samples, keeping the dataset bounded and biased toward more recent experience.
# 
# A lightweight PyTorch Dataset wrapper exposes the buffer in a format suitable for batching with a DataLoader, enabling standard supervised updates of the policy and value network.

# In[ ]:


class ReplayBuffer:
    # Simple replay buffer storing (X, π, z) training tuples collected from self-play.
    def __init__(self, max_size: int = 80_000):
        self.max_size = max_size
        self.X: List[np.ndarray] = []
        self.P: List[np.ndarray] = []
        self.Z: List[float] = []

    def add(self, X: List[np.ndarray], P: List[np.ndarray], Z: List[float]):
        assert len(X) == len(P) == len(Z)

        # compress in RAM (huge savings)
        X = [x.astype(np.float16, copy=False) for x in X]
        P = [p.astype(np.float16, copy=False) for p in P]

        self.X.extend(X)
        self.P.extend(P)
        self.Z.extend(Z)

        if len(self.X) > self.max_size:
            extra = len(self.X) - self.max_size
            self.X = self.X[extra:]
            self.P = self.P[extra:]
            self.Z = self.Z[extra:]

    def __len__(self) -> int:
        return len(self.X)


class BufferDataset(torch.utils.data.Dataset):
    # PyTorch Dataset wrapper around ReplayBuffer for use with DataLoader during training.
    def __init__(self, buf: ReplayBuffer):
        self.buf = buf  # Store a reference to the replay buffer (no data duplication).

    def __len__(self) -> int:
        # Dataset length equals the number of samples currently stored in the buffer.
        return len(self.buf)

    def __getitem__(self, idx: int):
        # Return one training sample (x, π, z) as PyTorch tensors.
        x = torch.from_numpy(self.buf.X[idx]).float()
        # Convert encoded state from numpy to float tensor.

        p = torch.from_numpy(self.buf.P[idx]).float()
        # Convert policy target π (dense vector over POLICY_SIZE) from numpy to float tensor.

        z = torch.tensor(self.buf.Z[idx], dtype=torch.float32)
        # Convert scalar value target (win/loss/draw) to float tensor.

        return x, p, z
        # Training tuple used by the loss: policy loss vs p and value loss vs z.


# ## Training step (KL policy + MSE value)
# 

# Network parameters are updated using supervised learning on batches sampled from the replay buffer.
# 
# The policy head is trained by minimizing the Kullback–Leibler divergence between the network’s predicted action distribution and the search-derived policy target. In parallel, the value head is trained using a mean squared error loss against the final game outcome.
# 
# The two losses are combined into a single objective and optimized using standard gradient-based methods, with gradient clipping applied for stability.

# In[ ]:


def train_steps(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    dl: torch.utils.data.DataLoader,
    device: str,
    steps: int = 200,
) -> Dict[str, float]:
    # Perform a fixed number of gradient update steps on batches sampled from the replay buffer.

    model.train()
    # Enable training mode (relevant for dropout/batchnorm, and standard practice during optimization).

    it = iter(dl)
    # Create an iterator over the DataLoader so we can manually draw batches with next().

    ema = {"loss": None, "lp": None, "lv": None}
    # Exponential moving averages of total loss, policy loss (lp), and value loss (lv) for stable logging.

    for _ in range(steps):
        # Run a fixed number of optimizer steps, restarting the DataLoader iterator as needed.
        try:
            x, p, z = next(it)
        except StopIteration:
            it = iter(dl)
            x, p, z = next(it)
        # If the iterator is exhausted, re-create it and continue sampling.

        x = x.to(device)
        p = p.to(device)
        z = z.to(device)
        # Move batch tensors to the same device as the model (CPU or GPU).

        opt.zero_grad(set_to_none=True)
        # Clear gradients from the previous step (set_to_none=True is a small performance optimization).

        logits, v_pred = model(x)
        # Forward pass: policy logits over the action space and scalar value predictions.

        logp = F.log_softmax(logits, dim=-1)
        # Convert logits into log-probabilities for numerically stable KL divergence computation.

        loss_policy = F.kl_div(logp, p, reduction="batchmean")
        # Policy loss: make the network imitate the search-improved policy target π from self-play.

        loss_value = F.mse_loss(v_pred, z)
        # Value loss: regress the predicted value toward the final game outcome z.

        loss = loss_policy + loss_value
        # Total loss (simple AlphaZero-style combination; can be weighted if needed).

        loss.backward()
        # Backpropagate gradients through the network.

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Gradient clipping for stability (prevents exploding gradients).

        opt.step()
        # Apply the optimizer update to the network parameters.

        for k, val in [("loss", loss.item()), ("lp", loss_policy.item()), ("lv", loss_value.item())]:
            # Track exponential moving averages for cleaner logging.
            if ema[k] is None:
                ema[k] = float(val)
            else:
                ema[k] = 0.98 * ema[k] + 0.02 * float(val)

    return {k: float(v) for k, v in ema.items()}
    # Return smoothed metrics that summarize training progress over the last steps.


# ## Checkpoint helpers
# 

# To support long-running experiments and allow training to be resumed across sessions, helper functions are provided to save and load model checkpoints.
# 
# Each checkpoint stores the network parameters, optimizer state, and basic training metadata, ensuring that training can be restarted consistently without loss of information.

# In[
def save_checkpoint(path, model, opt, step, extra=None):
    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": int(step),
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)  # rename atomico (POSIX)


def load_checkpoint(
    path: str,
    model: nn.Module,
    opt: Optional[torch.optim.Optimizer] = None
) -> dict:
    # Load a training checkpoint and restore model (and optionally optimizer) state.
    ckpt = torch.load(path, map_location="cpu")
    # Load onto CPU for portability; the model can be moved to GPU afterwards if needed.

    model.load_state_dict(ckpt["model"])
    # Restore the network parameters from the checkpoint.

    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
        # Restore optimizer state (important to continue training without losing momentum/Adam statistics).

    return ckpt
    # Return the loaded checkpoint dict (contains step counter and any extra metadata)

# ## One self-play game (ReconChess) → (X, P, Z)
# 

# A single self-play game is executed by running two instances of the same ReconChess player against each other using the local game runner.
# 
# During the game, each player records training samples (X,π) at decision time, where X is the encoded state and π is derived from the search visit counts. After the game ends, the final outcome is converted into a value target z and assigned to all samples collected by each player.
# 
# The resulting lists (X,P,Z) provide one complete episode of training data that can be appended to the replay buffer.

# In[ ]:


def selfplay_one_game(
    model: nn.Module,
    cfg: FastBotConfig,
    seed: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], Optional[Color], str]:
    # Play a single self-play game (model vs itself) and collect training samples (X, π, z).
    play_local_game = try_discover_play_local_game()
    # Obtain the local ReconChess runner function in a version-robust way.

    bot_w = RBCFastAZPlayer(model=model, cfg=cfg, seed=seed)
    bot_b = RBCFastAZPlayer(model=model, cfg=cfg, seed=seed + 1)
    # Create two players using the same network/config; use different RNG seeds to avoid identical randomness.

    bot_w.store_training = True
    bot_b.store_training = True
    # Enable in-game logging of (state X, policy target π, value target z) inside each bot.

    winner_color, reason, history = play_local_game(bot_w, bot_b)
    # Run one full game and collect winner/termination reason (history can be used for debugging if needed).

    X = bot_w.X + bot_b.X
    P = bot_w.P + bot_b.P
    Z = bot_w.Z + bot_b.Z
    # Merge samples produced by both players into a single dataset for training.

    # Safety fallback: if value targets were not filled (e.g., handle_game_end not triggered),
    # reconstruct z targets from the final winner and assign them to each player's samples.
    if len(Z) != len(X):
        if winner_color is None:
            z_w = 0.0  # Draw/unknown result.
        else:
            z_w = 1.0 if bool(winner_color) == chess.WHITE else -1.0
            # Outcome from White's perspective: White win -> +1, White loss -> -1.

        z_b = -z_w
        # Outcome from Black's perspective is the opposite of White's.

        Z = [z_w] * len(bot_w.X) + [z_b] * len(bot_b.X)
        # Assign the final outcome to all states generated by each player (AlphaZero-style episodic value target).

    return X, P, Z, winner_color, reason
    # Return collected training samples and basic game outcome info for logging.


# ## Self-play → train → eval loop (FAST)
# 

# The main training loop alternates between data generation and network updates.
# 
# At each iteration, a small batch of self-play games is generated to produce new (X,π,z) samples, which are appended to the replay buffer. The policy/value network is then updated for a fixed number of gradient steps using mini-batches sampled from the buffer.
# 
# After training, the current model is evaluated in a short match series against a simple baseline opponent to provide a quick progress signal, and a checkpoint plus a CSV log entry are saved for later inspection

# In[ ]:
#from src.internal_elo_eval import InternalEloState, eval_and_update
#from src.elo import EloConfig


def play_games_between_checkpoints(ckpt_a_path: str, ckpt_b_path: str, n_games: int):
    wins = losses = draws = 0
    play_local_game = try_discover_play_local_game()

    # carica modello A
    model_a = FastPolicyValueNet(in_ch=15, trunk=64).to(DEVICE)
    opt_dummy = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    load_checkpoint(ckpt_a_path, model_a, opt_dummy)

    # carica modello B
    model_b = FastPolicyValueNet(in_ch=15, trunk=64).to(DEVICE)
    opt_dummy2 = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    load_checkpoint(ckpt_b_path, model_b, opt_dummy2)

    for g in range(n_games):
        bot_a = RBCFastAZPlayer(model=model_a, cfg=FastBotConfig(sims=20), seed=g)
        bot_b = RBCFastAZPlayer(model=model_b, cfg=FastBotConfig(sims=20), seed=g+999)

        if g % 2 == 0:
            winner_color, _, _ = play_local_game(bot_a, bot_b)
            a_is_white = True
        else:
            winner_color, _, _ = play_local_game(bot_b, bot_a)
            a_is_white = False

        if winner_color is None:
            draws += 1
        else:
            a_won = (bool(winner_color) == chess.WHITE) if a_is_white else (bool(winner_color) == chess.BLACK)
            if a_won:
                wins += 1
            else:
                losses += 1

    return wins, losses, draws

def run_selfplay_training(
    iters: int = 5,
    games_per_iter: int = 6,
    train_steps_per_iter: int = 300,
    batch_size: int = 64,
    sims_selfplay: int = 40,
    sims_eval: int = 20,
    eval_games: int = 50,
    save_every: int = 250,
    keep_last: int = 3,
    seed: int = 0,
    out_dir: str = "fast_checkpoints",
    results_csv: str = "results.csv",
):
    
    # Main AlphaZero-like loop: self-play -> add to replay buffer -> train -> quick eval -> save checkpoint -> log CSV.

    os.makedirs(out_dir, exist_ok=True)
    # Ensure output directory exists for checkpoints and logs.

    results_path = os.path.join(out_dir, results_csv)
    # Store results CSV inside the same output directory for convenience.

    # Initialize CSV header once (if the file does not exist yet).
    fieldnames = [
        "timestamp", "iter", "buffer_size",
        "loss", "loss_policy", "loss_value",
        "eval_wins", "eval_losses", "eval_draws",

        # ---- NEW INTERNAL ELO ----
        "elo_internal",
        "int_vs_prev_score",
        "int_vs_prev_elo_delta",

        "sims_selfplay", "sims_eval",
        "games_per_iter", "train_steps_per_iter", "batch_size",
        "eval_games",
        "ckpt_path",
    ]
    if not os.path.exists(results_path):
        with open(results_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        # Create the CSV file with a header row the first time we run the loop.

    model = FastPolicyValueNet(in_ch=15, trunk=64).to(DEVICE)
    # Create the policy/value network used for both self-play and training.

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # Optimizer for training the network parameters.

    buf = ReplayBuffer(max_size=200_000)
    # Replay buffer storing training samples (X, π, z) collected from self-play.

    step = 0
    # Global step counter for checkpoint bookkeeping (can represent training iterations or updates).

    AGENT_ID = "agent"
    elo_state = InternalEloState(ratings={AGENT_ID: 1000.0}, cfg=EloConfig(k=32.0))
    for it in range(iters):
        # Each iteration generates new self-play data, trains the network, evaluates, and checkpoints.

        cfg = FastBotConfig(sims=sims_selfplay, c_puct=1.5)
        # Configuration for self-play search (more sims generally yields stronger targets but slower games).

        # 1) Self-play data collection
        for g in range(games_per_iter):
            # Play one self-play game and add all collected samples to the replay buffer.
            X, P, Z, winner, reason = selfplay_one_game(model, cfg, seed=seed + it * 1000 + g)
            buf.add(X, P, Z)

        # 2) Training on replay buffer
        ds = BufferDataset(buf)
        # Wrap replay buffer as a PyTorch Dataset.

        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
        )
        # DataLoader provides shuffled mini-batches for training.

        metrics = train_steps(model, opt, dl, device=DEVICE, steps=train_steps_per_iter)
        # Perform several gradient steps and return smoothed losses.

        # 3) Quick evaluation vs RandomBot (ALWAYS)
        try:
            wins = 0
            losses = 0
            draws = 0
            play_local_game = try_discover_play_local_game()

            for eg in range(eval_games):
                bot = RBCFastAZPlayer(
                    model=model,
                    cfg=FastBotConfig(sims=sims_eval, c_puct=1.5),
                    seed=seed + 999 + it * 10 + eg,
                )
                opp = SimpleRandomBot()

                winner_color, reason, history = play_local_game(bot, opp)

                if winner_color is None:
                    draws += 1
                elif bool(winner_color) == chess.WHITE:
                    wins += 1
                else:
                    losses += 1

            eval_res = {"wins": wins, "losses": losses, "draws": draws}

        except Exception as e:
            print("Eval failed:", repr(e))
            eval_res = {"wins": 0, "losses": 0, "draws": 0}

        # 4) Save checkpoint (disk-safe)
        latest_path = os.path.join(out_dir, "latest.pt")
        save_checkpoint(
            latest_path,
            model,
            opt,
            step,
            extra={"iter": it, "buffer": len(buf)},
        )

        ckpt_path = ""  # only sometimes we write a full ckpt
        if (save_every is not None) and (save_every > 0) and (it % save_every == 0):
            ckpt_path = os.path.join(out_dir, f"ckpt_iter_{it}.pt")
            save_checkpoint(
                ckpt_path,
                model,
                opt,
                step,
                extra={"iter": it, "buffer": len(buf)},
            )

            # keep only last K checkpoints
            if keep_last is not None and keep_last > 0:
                import glob, re
                paths = glob.glob(os.path.join(out_dir, "ckpt_iter_*.pt"))

                def itnum(p):
                    m = re.search(r"ckpt_iter_(\d+)\.pt$", p)
                    return int(m.group(1)) if m else -1

                paths.sort(key=itnum)
                for p in paths[:-keep_last]:
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        print(f"[iter {it}] buffer={len(buf)} train={metrics} eval6={eval_res} latest={latest_path} ckpt={ckpt_path}")

        # ---- INTERNAL ELO (cumulative vs fixed RandomBot=1000) ----
        wins_rb = eval_res.get("wins", 0)
        losses_rb = eval_res.get("losses", 0)
        draws_rb = eval_res.get("draws", 0)

        n_eval = wins_rb + losses_rb + draws_rb
        r_agent = elo_state.ratings.get(AGENT_ID, 1000.0)

        if n_eval == 0:
            # Eval non valida -> non toccare Elo
            current_elo = r_agent
            elo_score = None
            elo_delta = 0.0
        else:
            s_a = score_from_counts(wins_rb, losses_rb, draws_rb)

            r_baseline = 1000.0  # RandomBot fixed
            e_a = expected_score(r_agent, r_baseline, elo_state.cfg)
            delta = elo_state.cfg.k * (s_a - e_a)

            r_agent_new = r_agent + delta
            elo_state.ratings[AGENT_ID] = r_agent_new

            current_elo = r_agent_new
            elo_score = s_a
            elo_delta = delta

        # 5) Append one row to results CSV
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "iter": it,
            "buffer_size": len(buf),
            "loss": metrics.get("loss"),
            "loss_policy": metrics.get("lp"),
            "loss_value": metrics.get("lv"),
            "eval_wins": eval_res.get("wins", 0),
            "eval_losses": eval_res.get("losses", 0),
            "eval_draws": eval_res.get("draws", 0),
            "sims_selfplay": sims_selfplay,
            "sims_eval": sims_eval,
            "games_per_iter": games_per_iter,
            "train_steps_per_iter": train_steps_per_iter,
            "batch_size": batch_size,
            "eval_games": eval_games,
            "ckpt_path": ckpt_path,
            "elo_internal": current_elo,
            "int_vs_prev_score": elo_score,
            "int_vs_prev_elo_delta": elo_delta,
        }

        with open(results_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)
        # Log metrics and eval results to CSV for later plotting/analysis.

        step += 1
        # Increment checkpoint/training iteration counter.

    return model
    # Return the final trained model after all iterations.


# ### Run a small self-play training (start tiny)
# 

# In[ ]:


# model = run_selfplay_training(
#     iters=3,
#     games_per_iter=2,
#     train_steps_per_iter=200,
#     batch_size=64,
#     sims_selfplay=20,
#     sims_eval=10,
#     seed=0,
#     out_dir="fast_checkpoints",
# )


# ## Results plots
# After training, plot loss and winrate vs random.
# 

# In[ ]:

def plot_results(csv_path: str, out_dir: str = "."):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    # robust sorting
    df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
    df = df.dropna(subset=["iter"]).sort_values("iter")

    os.makedirs(out_dir, exist_ok=True)

    # Global style (minimal, thesis-friendly)
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
    })

    x = df["iter"].to_numpy()

    # ===============================
    # LOSSES (3 panels)
    # ===============================
    fig, axs = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axs[0].plot(x, df["loss_policy"], marker="o")
    axs[0].set_ylabel("Policy loss")
    axs[0].set_title("Policy loss")

    axs[1].plot(x, df["loss_value"], marker="o")
    axs[1].set_ylabel("Value loss")
    axs[1].set_title("Value loss")

    axs[2].plot(x, df["loss"], marker="o")
    axs[2].set_ylabel("Total loss")
    axs[2].set_xlabel("Iteration")
    axs[2].set_title("Total loss")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "losses.png"))
    plt.savefig(os.path.join(out_dir, "losses.pdf"))
    plt.close(fig)

    # ===============================
    # WINRATE + 95% CI (Wilson)
    # ===============================
    wins = df["eval_wins"].to_numpy()
    losses = df["eval_losses"].to_numpy()
    draws = df["eval_draws"].to_numpy()

    n = (wins + losses + draws).clip(min=1)

    # winrate puro
    p = wins / n
    # Moving average (rolling mean) over iterations
    winrate_ma = pd.Series(p).rolling(window=3, min_periods=1).mean().to_numpy()

    z = 1.96
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2*n)) / denom
    half = (z / denom) * np.sqrt((p*(1 - p) / n) + (z**2) / (4*(n**2)))

    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)

    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(x, p, marker="o", label="Winrate")
    plt.plot(x, winrate_ma, linestyle="--", label="Winrate (MA, w=3)")
    plt.fill_between(x, lo, hi, alpha=0.25, label="95% CI (Wilson)")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Iteration")
    plt.ylabel("Winrate vs RandomBot")
    plt.title("Evaluation performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "winrate.png"))
    plt.savefig(os.path.join(out_dir, "winrate.pdf"))
    plt.close(fig)

    # ===============================
    # INTERNAL ELO
    # ===============================
    if "elo_internal" in df.columns:
        elo = df["elo_internal"].to_numpy()
        fig = plt.figure(figsize=(7, 4.5))
        plt.plot(x, elo, marker="o", label="Internal Elo")
        # padding automatico
        ymin = float(np.min(elo))
        ymax = float(np.max(elo))
        pad = max(5.0, 0.05 * (ymax - ymin + 1e-9))
        plt.ylim(ymin - pad, ymax + pad)
        plt.xlabel("Iteration")
        plt.ylabel("Elo (internal)")
        plt.title("Internal Elo progress")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "elo.png"))
        plt.savefig(os.path.join(out_dir, "elo.pdf"))
        plt.close(fig)

    # ===============================
    # BUFFER SIZE
    # ===============================
    if "buffer_size" in df.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df["iter"], df["buffer_size"], marker="o", label="Buffer size")
        plt.xlabel("Iteration")
        plt.ylabel("Samples in replay buffer")
        plt.title("Replay buffer growth")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "buffer_size.png"), dpi=200)
        plt.savefig(os.path.join(out_dir, "buffer_size.pdf"))
        plt.close()

    print("Saved scientific plots.")

'''Example:
df = plot_results("fast_checkpoints/results.csv")'''


# This notebook provides a complete and executable reference implementation of a learning-based RBC agent, suitable for experimentation and further extensions.
