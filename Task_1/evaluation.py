from ranx import Qrels, Run, evaluate
import sys

def main():
    file_name = sys.argv[1]
    qrels = Qrels.from_file("simpletext_2024_task1_train.qrels", kind="trec")
    print(qrels.size)
    run = Run.from_file(file_name, kind="trec")

    print(evaluate(qrels, run, "ndcg@10", make_comparable=True))
    print(evaluate(qrels, run, "map", make_comparable=True))

if __name__ == '__main__':
    main()
