from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)

    trainer = Trainer(args)
    if args.model_epoch_start > 0:
        trainer.load(args.model_epoch_start)

    if args.model_train == 1:
        for epoch in range(args.model_epoch_start, args.model_epoch_end):
            trainer.cur_epoch = epoch
            trainer.fit()  # 训练
            trainer.save(epoch + 1)
            #trainer.score('val')
            trainer.score('test')  # 测试
            #if not args.demo:
             #   trainer.score('test2')
    elif args.model_train == 0:  # case study
        trainer.cur_epoch = args.model_epoch_start
        trainer.score_rank('test')
        # trainer.batch_score('test', test_k=100)
    else:
        trainer.cur_epoch = args.model_epoch_start
        trainer.score('test')
        """
        test_matching = True
        trainer.cur_epoch = args.model_epoch_start
        #trainer.score('val', test_matching=test_matching)
        trainer.score('test', test_matching=test_matching)
        #if not args.demo:
         #   trainer.score('test2')
        """

if __name__ == "__main__":
    main()
