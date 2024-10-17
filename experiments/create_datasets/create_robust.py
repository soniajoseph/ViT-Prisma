import time

from vit_prisma.utils.data_utils.cifar.cifar_10_utils import load_tinyimagenet


def single_pgd_step_robust(model, X, y, alpha, delta):
    with tf.GradientTape() as tape:
        tape.watch(delta)
        loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )(y, model(X + delta))  # comparing to robust model representation layer

    grad = tape.gradient(loss, delta)
    normgrad = tf.reshape(norm(grad), (-1, 1, 1, 1))
    delta -= alpha * grad / (normgrad + 1e-10)  # normalized gradient step
    delta = tf.math.minimum(tf.math.maximum(delta, -X), 1 - X)  # clip X+delta to [0,1]
    return delta, loss


def pgd_l2_robust(model, X, y, alpha, num_iter, epsilon=0, example=False):
    delta = tf.zeros_like(X)
    loss = 0
    fn = tf.function(single_pgd_step_robust)
    for t in range(num_iter):
      delta, loss = fn(model, X, y, alpha, delta)
    # Prints out loss to evaluate if it's actually learning (currently broken)
    if example:
        print(f'{num_iter} iterations, final MSE {loss}')
    return delta


def robustify(robust_mod, train_ds, iters=1000, alpha=0.1, batch_size=BATCH_SIZE):
    robust_train = []
    orig_labels = []
    example = False

    train_to_pull = list(iter(train_ds))
    start_rn = np.random.randint(0, len(train_ds))
    rand_batch = train_to_pull[start_rn][0]

    start_time = time.time()
    for i, (img_batch, label_batch) in enumerate(train_ds):
        inter_time = time.time()

        # For the last batch, it is smaller than batch_size and thus we match the size for the batch of initial images
        if img_batch.shape[0] < batch_size:
            rand_batch = rand_batch[:img_batch.shape[0]]

        # Get the goal representation
        goal_representation = robust_mod(img_batch)

        # Upate the batch of images
        learned_delta = pgd_l2_robust(robust_mod, rand_batch, goal_representation, alpha=alpha, num_iter=iters)
        robust_update = (rand_batch + learned_delta)

        # Add the updated images and labels to their respective lists
        robust_train.append(robust_update)
        orig_labels.append(label_batch)

        # Measure the time
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            elapsed_tracking = time.time() - inter_time
            print(
                f'Robustified {(i + 1) * batch_size} images in {elapsed:0.3f} seconds; Took {elapsed_tracking:0.3f} seconds for this particular iteration')

            # Reset random image batch
        rn = np.random.randint(0, len(train_ds) - 1)  # -1 because last batch might be smaller
        rand_batch = train_to_pull[rn][0]

    return robust_train, orig_labels

if __name__ == "__main__":
    
    train_dataset, val_dataset, test_dataset = load_tinyimagenet("tinyimagenet")
    robust_train, orig_labels = robustify(robustifier, train_dataset, iters=1000, alpha=0.1)

    # Print out the shapes
    print(tf.concat(robust_train, axis=0).shape)
    print(tf.concat(orig_labels, axis=0).shape)