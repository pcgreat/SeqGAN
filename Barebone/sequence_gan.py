import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import tensorflow as tf
import logging
import random
import pickle
from Barebone.dataloader import Gen_Data_loader, Dis_dataloader
from Barebone.generator import Generator
from Barebone.discriminator import Discriminator
from Barebone.rollout import ROLLOUT

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32  # embedding dimension
HIDDEN_DIM = 32  # hidden state dimension of lstm cell
SEQ_LENGTH = 30  # sequence length
START_TOKEN = 0
PRE_GENERATOR_EPOCH_NUM = 120  # supervise (maximum likelihood estimation) epochs
PRE_DISCRIMINATOR_EPOCH_NUM = 50  # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]  # , 20, 32]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]  # , [160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
positive_file = 'data/realtrain_essay_train.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'data/realtrain_essay_eval.txt'
vocab_file = "data/vocab_essay.pkl"
generated_num = 10000

# create logger with 'spam_application'
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('save/experiment.log')
ch = logging.StreamHandler()
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def pretrain_target_loss(sess, trainable_model, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    pretrain_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        pretrain_loss = trainable_model.pretrain_step_eval(sess, batch)
        pretrain_losses.append(pretrain_loss)
    return np.mean(pretrain_losses)


def target_loss(sess, trainable_model, data_loader, rollout, discriminator):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    pretrain_losses = []
    g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        rewards = rollout.get_reward(sess, batch, 16, discriminator)
        g_loss, pretrain_loss = trainable_model.step_eval(sess, batch, rewards)
        g_losses.append(g_loss)
        pretrain_losses.append(pretrain_loss)
    return np.mean(g_losses), np.mean(pretrain_losses)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)  # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE, SEQ_LENGTH)

    word, vocab = pickle.load(open(vocab_file, "rb"))
    vocab_size = len(vocab)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size,
                                  embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda)
    rollout = ROLLOUT(generator, 0.8)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    # generate_samples(sess, generator, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    #  pre-train generator
    logger.info('Start pre-training generator...')
    for epoch in range(PRE_GENERATOR_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)
        # logger.info("epoch %s: training_nll_loss: %s" % (epoch, loss))
        if epoch % 5 == 0:
            # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = pretrain_target_loss(sess, generator, likelihood_data_loader)
            logger.info(('pre-train generator: epoch', epoch, 'training_nll_loss', loss, "test_nll_loss", test_loss))

    logger.info('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for epoch in range(PRE_DISCRIMINATOR_EPOCH_NUM):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        D_losses = []
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                D_loss, _ = sess.run([discriminator.loss, discriminator.train_op], feed)
                D_losses.append(D_loss)
        logger.info(("pre-train discriminator: epoch", epoch, "training_loss", np.mean(D_losses)))

    logger.info('#########################################################################')
    logger.info('Start Adversarial Training...')
    for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            _, g_loss, pretrain_loss = sess.run([generator.g_updates, generator.g_loss, generator.pretrain_loss],
                                                feed_dict=feed)
            logger.info(('total_batch: ', total_batch, "train_g_loss:", g_loss, "train_nll_loss:", pretrain_loss))

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, "./save/coco_" + str(total_batch) + ".txt")
            # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_g_loss, test_pretrain_loss = target_loss(sess, generator, likelihood_data_loader, rollout,
                                                          discriminator)
            logger.info(
                ('total_batch: ', total_batch, "test_g_loss:", test_g_loss, "test_nll_loss:", test_pretrain_loss))

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            D_losses = []
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    D_loss, _ = sess.run([discriminator.loss, discriminator.train_op], feed)
                    D_losses.append(D_loss)
            logger.info(("discriminator: total_batch", total_batch, "training_loss", np.mean(D_losses)))


if __name__ == '__main__':
    main()
