
import sys, argparse
from pp14_ddpg.andes_replay import replay_with_andes

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=str, help="Path to exported trace CSV")
    ap.add_argument("--andes-case", type=str, default="", help="Path to ANDES IEEE-14 case")
    ap.add_argument("--out", type=str, default="andes_replay")
    args = ap.parse_args()
    replay_with_andes(args.trace, args.andes_case, out_prefix=args.out)
