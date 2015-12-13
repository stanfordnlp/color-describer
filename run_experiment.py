from bt import evaluate, metrics, output
import learners
import color_instances


def main():
    learner = learners.LookupLearner()

    train_data = color_instances.get_training_instances()
    learner.train(train_data)

    train_results = evaluate.evaluate(learner, train_data, metrics.log_likelihood)
    output.output_results(train_results, 'train')

    dev_data = color_instances.get_dev_instances()
    dev_results = evaluate.evaluate(learner, dev_data, metrics.log_likelihood)
    output.output_results(dev_results, 'dev')


if __name__ == '__main__':
    main()
