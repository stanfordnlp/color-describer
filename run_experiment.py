from bt import evaluate, metrics, output, timing, config
import datetime
import learners
import color_instances


def main():
    config.options()
    learner = learners.HistogramLearner()

    timing.set_resolution(datetime.timedelta(seconds=1))
    timing.start_task('Step', 4)

    timing.progress(0)
    train_data = color_instances.get_training_instances()[:1000]

    timing.progress(1)
    learner.train(train_data)

    timing.progress(2)
    train_results = evaluate.evaluate(learner, train_data, metrics.log_likelihood)
    output.output_results(train_results, 'train')

    timing.progress(3)
    dev_data = color_instances.get_dev_instances()[:100]
    dev_results = evaluate.evaluate(learner, dev_data, metrics.log_likelihood)
    output.output_results(dev_results, 'dev')

    timing.end_task()


if __name__ == '__main__':
    main()
