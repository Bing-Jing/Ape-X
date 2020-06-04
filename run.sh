#!/bin/bash
tmux new -s replay -d 'python3 replay.py; read'
tmux new -s learner -d 'REPLAY_IP="127.0.0.1" N_ACTORS=1 python3 learner.py --cuda; read'
tmux new -s actor0 -d 'REPLAY_IP="127.0.0.1" LEARNER_IP="127.0.0.1" ACTOR_ID=0 N_ACTORS=1 python3 actor.py; read'
tmux new -s evaluator -d 'LEARNER_IP="127.0.0.1" python3 eval.py; read'
