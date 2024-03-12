cd experiments/'Overall Performance'
git status
git add
git commit
git push origin master


python src/main.py --model-epoch-start 0 --model-epoch-end 2 --model-train 1 --demo
python src/main.py --model-epoch-start 0 --model-epoch-end 20 --model-train 1


python src/main.py --model-name MyGNN --model-epoch-start 0 --model-epoch-end 2 --model-train 1 --demo

nohup python src/main.py --model-epoch-start 0 --model-epoch-end 2 --model-train 1 --demo > log/demo.log 2>&1 &