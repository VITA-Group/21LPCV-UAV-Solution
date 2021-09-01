python main.py --source inputs/4p1b_01A2/4p1b_01A2.m4v --groundtruth inputs/4p1b_01A2/4p1b_01A2_init.csv --skip-frames 10
python main.py --source inputs/5p2b_01A1/5p2b_01A1.m4v --groundtruth inputs/5p2b_01A1/5p2b_01A1_init.csv --skip-frames 10
python main.py --source inputs/5p4b_01A2/5p4b_01A2.m4v --groundtruth inputs/5p4b_01A2/5p4b_01A2_init.csv --skip-frames 10
python main.py --source inputs/5p5b_03A1/5p5b_03A1.m4v --groundtruth inputs/5p5b_03A1/5p5b_03A1_init.csv --skip-frames 10
python main.py --source inputs/7p3b_02M/7p3b_02M.m4v --groundtruth inputs/7p3b_02M/7p3b_02M_init.csv --skip-frames 10
python evaluate_submission.py --submitted 4p1b_01A2
python evaluate_submission.py --submitted 5p2b_01A1
python evaluate_submission.py --submitted 5p4b_01A2
python evaluate_submission.py --submitted 5p5b_03A1
python evaluate_submission.py --submitted 7p3b_02M
