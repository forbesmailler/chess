# constants.py
files  = "abcdefgh"
ranks  = "12345678"
ALL_UCIS = []
for f1 in files:
    for r1 in ranks:
        for f2 in files:
            for r2 in ranks:
                u = f1 + r1 + f2 + r2
                ALL_UCIS.append(u)
                # promotions on the last rank for white (8) or black (1)
                if r2 in ("8", "1"):
                    for p in ("q","r","b","n"):
                        ALL_UCIS.append(u + p)
# dedupe
ALL_UCIS = list(dict.fromkeys(ALL_UCIS))
UCI_TO_IDX = {u:i for i,u in enumerate(ALL_UCIS)}
NUM_ACTIONS = len(ALL_UCIS)
