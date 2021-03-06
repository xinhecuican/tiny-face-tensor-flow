.RECIPEPREFIX +=

PYTHON=./venv/Scripts/python
ROOT=data/WIDER
TRAINDATA=$(ROOT)/wider_face_split/wider_face_train_bbx_gt.txt
VALDATA=$(ROOT)/wider_face_split/wider_face_val_bbx_gt.txt
TESTDATA=$(ROOT)/wider_face_split/wider_face_test_filelist.txt
EPOCH=60
CHECKPOINT=weights/edge_checkpoint_50.pth
WORKERS = 6
BATCH_SIZE = 4

main:
	$(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT)

resume: 
	$(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --resume $(CHECKPOINT) --batch_size $(BATCH_SIZE) --workers $(WORKERS) --epochs $(EPOCH)

evaluate: 
	$(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split val

evaluation:
	$(PYTHON) eval_tools/wider_eval.py

test: 
	$(PYTHON) evaluate.py $(TESTDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split test

cluster: 
	cd utils; $(PYTHON) cluster.py $(TRAIN_INSTANCES)

debug: 
	$(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --batch_size 1 --workers 0 --debug

debug-evaluate: 
	$(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split val --batch_size 1 --workers 0 --debug
