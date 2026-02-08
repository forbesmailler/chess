# Handcrafted Evaluation

The handcrafted eval scores a position in centipawns from white's perspective using **tapered evaluation** — separate middlegame (MG) and endgame (EG) scores blended by a game-phase weight.

## Tapered Eval

Every evaluation term produces both an MG and an EG score. At the end, the two are blended:

```
eval = (mg_score * mg_phase + eg_score * eg_phase) / TOTAL_PHASE
```

The **phase** is computed by summing piece weights still on the board:

| Piece  | Phase weight |
|--------|-------------|
| Pawn   | 0           |
| Knight | 1           |
| Bishop | 1           |
| Rook   | 2           |
| Queen  | 4           |
| King   | 0           |

Total phase at game start = 4(1) + 4(1) + 4(2) + 2(4) = **24**. As pieces are captured, phase decreases and the EG score gains influence.

## Material

Each piece type has separate MG and EG values (centipawns):

| Piece  | MG  | EG  |
|--------|-----|-----|
| Pawn   | 100 | 110 |
| Knight | 320 | 310 |
| Bishop | 330 | 330 |
| Rook   | 500 | 520 |
| Queen  | 900 | 950 |

## Piece-Square Tables

Per-square bonuses for each piece type, encouraging good piece placement. Separate MG and EG tables exist for pawns and kings; other pieces share the same table across both phases.

Key ideas encoded in the tables:
- **Pawns**: MG penalizes unmoved center pawns, encourages center control. EG rewards advancement (up to +80 on the 7th rank).
- **Knights**: Centralized knights get up to +20; corner/edge knights penalized up to -50.
- **Bishops**: Prefer diagonals and avoid edges.
- **Rooks**: Slight bonus on the 7th rank (+5–10); otherwise neutral.
- **Queen**: Prefers center, penalized on edges.
- **King MG**: Stay castled — corners get +20/+30, center penalized up to -50.
- **King EG**: Opposite — centralized king gets +40, corners penalized -50.

Black's PST values are looked up by vertically mirroring the square (`sq ^ 56`).

## Pawn Structure

Three structural penalties/bonuses, evaluated per pawn:

### Passed Pawns

A pawn with no enemy pawns on the same or adjacent files that could block or capture it. Bonus scales quadratically with advancement:

```
bonus = 10 + distance_from_start^2 * 3
```

Applied at half strength in MG, full strength in EG (passed pawns matter more in endgames).

### Isolated Pawns

A pawn with no friendly pawns on adjacent files. Penalty: **-15 MG, -20 EG**.

### Doubled Pawns

More than one friendly pawn on the same file. Penalty per pawn: **-10 MG, -15 EG**.

## Rook on Open Files

Per rook on the same file:

| File type | MG  | EG  |
|-----------|-----|-----|
| Open (no pawns) | +15 | +10 |
| Semi-open (no own pawns) | +8 | +5 |

## Bishop Pair

Having two or more bishops: **+30 MG, +50 EG**. Evaluated independently for each side.

## Mobility

For knights, bishops, rooks, and queens: count the number of squares attacked (excluding squares occupied by own pieces), multiplied by a per-piece bonus:

| Piece  | Bonus per square |
|--------|-----------------|
| Knight | 4               |
| Bishop | 3               |
| Rook   | 2               |
| Queen  | 1               |

Applied equally in MG and EG.

## King Pawn Shield

MG-only bonus. Checks the three files around the king (king file and adjacent files) for friendly pawns on the 1st or 2nd rank ahead of the king. Each shielding pawn: **+10 MG**.

## Terminal Positions

- **Checkmate**: returns +/- 10000 (from white's perspective)
- **Stalemate or draw**: returns 0
