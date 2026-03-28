import time
import pandas as pd


# ── Board Representation ──────────────────────────────────────────────────────

def cross(A, B):
    """Cross product of elements in A and elements in B."""
    return [a + b for a in A for b in B]

digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits
squares  = cross(rows, cols)

unitlist = (
    [cross(rows, c) for c in cols] +           # columns
    [cross(r, cols) for r in rows] +            # rows
    [cross(rs, cs)                              # boxes
     for rs in ('ABC', 'DEF', 'GHI')
     for cs in ('123', '456', '789')]
)

units = dict((s, [u for u in unitlist if s in u]) for s in squares)
peers = dict((s, set(sum(units[s], [])) - {s})   for s in squares)


# ── Parsing ───────────────────────────────────────────────────────────────────

def grid_values(grid):
    """Convert an 81-char string into a dict of {square: char}.
    Accepts '0' or '.' for empty cells."""
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81, f"Expected 81 chars, got {len(chars)}"
    return dict(zip(squares, chars))


def parse_grid(grid):
    """Convert grid to a dict of possible values {square: digits}.
    Returns False if a contradiction is detected during propagation."""
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False
    return values


# ── Constraint Propagation ────────────────────────────────────────────────────

def assign(values, s, d):
    """Assign digit d to square s by eliminating all other digits.
    Returns values, or False if a contradiction is detected."""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    return False


def eliminate(values, s, d):
    """Eliminate digit d from values[s] and propagate constraints.

    Two propagation rules:
      1. If a square is reduced to one value, eliminate that value from peers.
      2. If a unit has only one place for a digit, assign it there.

    Returns values, or False if a contradiction is detected."""
    if d not in values[s]:
        return values  # already eliminated

    values[s] = values[s].replace(d, '')

    # Rule 1: naked single
    if len(values[s]) == 0:
        return False  # contradiction: no candidates left
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False

    # Rule 2: hidden single
    for u in units[s]:
        dplaces = [sq for sq in u if d in values[sq]]
        if len(dplaces) == 0:
            return False  # contradiction: no place for d in this unit
        elif len(dplaces) == 1:
            if not assign(values, dplaces[0], d):
                return False

    return values


# ── Search ────────────────────────────────────────────────────────────────────

def solve(grid):
    """Solve a sudoku puzzle given as an 81-char string."""
    return search(parse_grid(grid))


def search(values):
    """Depth-first search with constraint propagation.
    Picks the unsolved square with fewest candidates (MRV heuristic)."""
    if values is False:
        return False
    if all(len(values[s]) == 1 for s in squares):
        return values  # solved

    # minimum remaining values heuristic
    _, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)

    return some(
        search(assign(values.copy(), s, d))
        for d in values[s]
    )


def some(seq):
    """Return the first truthy element of seq, or False."""
    for e in seq:
        if e:
            return e
    return False


# ── Display ───────────────────────────────────────────────────────────────────

def display(values):
    """Print a sudoku values dict as a formatted 2D grid."""
    if values is False:
        print("No solution.")
        return
    width = 1 + max(len(values[s]) for s in squares)
    line  = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(
            values[r + c].center(width) + ('|' if c in '36' else '')
            for c in cols
        ))
        if r in 'CF':
            print(line)
    print()


def display_puzzle(grid):
    """Display a raw puzzle string as a formatted grid."""
    mapping = {s: c for s, c in zip(squares, grid)}
    display({s: ('.' if mapping[s] in '0.' else mapping[s]) for s in squares})


# ── Benchmarking ──────────────────────────────────────────────────────────────

def count_givens(puzzle):
    """Count the number of given (non-empty) cells in a puzzle string."""
    return sum(1 for c in puzzle if c not in '0.')


def count_ambiguous(puzzle):
    """Count squares still ambiguous after constraint propagation."""
    parsed = parse_grid(puzzle)
    if parsed is False:
        return -1  # contradiction detected
    return sum(1 for s in squares if len(parsed[s]) > 1)


def benchmark(puzzles, solutions, n=1000):
    """Benchmark the solver on n puzzles. Reports accuracy and timing."""
    correct = 0
    times   = []

    for i in range(n):
        start   = time.time()
        result  = solve(puzzles[i])
        elapsed = time.time() - start
        times.append(elapsed)

        if result:
            solved = ''.join(result[s] for s in squares)
            if solved == solutions[i]:
                correct += 1

    avg_ms = sum(times) / len(times) * 1000
    max_ms = max(times) * 1000

    print(f"Accuracy : {correct}/{n}")
    print(f"Avg time : {avg_ms:.2f}ms")
    print(f"Max time : {max_ms:.2f}ms")

    return times


def find_hardest(puzzles, n=1000):
    """Return the index and solve time of the hardest puzzle in the first n."""
    times = []
    for i in range(n):
        start   = time.time()
        solve(puzzles[i])
        elapsed = time.time() - start
        times.append((elapsed, i))
    times.sort(reverse=True)
    return times[0]  # (time, index)


# ── Known Hard Puzzles ────────────────────────────────────────────────────────

HARD_PUZZLES = {
    'hard1'   : '.....6....59.....82....8....45........3........6..3.54...325..6..................',
    'hardest' : '85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4.',
}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Load dataset
    print("Loading dataset...")
    df       = pd.read_csv('sudoku.csv', dtype=str)
    puzzles  = df['quizzes'].tolist()
    solutions = df['solutions'].tolist()
    print(f"Loaded {len(df):,} puzzles\n")

    # Benchmark on easy puzzles
    print("=== Benchmark: 1000 puzzles from dataset ===")
    benchmark(puzzles, solutions, n=1000)
    print()

    # Check how many require search
    requires_search = sum(
        1 for i in range(1000)
        if count_ambiguous(puzzles[i]) > 0
    )
    print(f"Puzzles requiring search: {requires_search}/1000\n")

    # Hardest from dataset
    elapsed, idx = find_hardest(puzzles, n=1000)
    print(f"=== Hardest in sample: index {idx}, {elapsed*1000:.2f}ms ===")
    print(f"Givens: {count_givens(puzzles[idx])}, "
          f"Ambiguous after propagation: {count_ambiguous(puzzles[idx])}")
    display(solve(puzzles[idx]))

    # Known hard puzzles
    print("=== Known hard puzzles ===")
    for name, puzzle in HARD_PUZZLES.items():
        parsed = parse_grid(puzzle)
        if parsed is False:
            print(f"{name}: invalid puzzle")
            continue
        ambiguous = count_ambiguous(puzzle)
        start     = time.time()
        result    = solve(puzzle)
        elapsed   = time.time() - start
        print(f"{name}: {count_givens(puzzle)} givens, "
              f"{ambiguous} ambiguous after propagation, "
              f"solved in {elapsed*1000:.2f}ms")
        display(result)