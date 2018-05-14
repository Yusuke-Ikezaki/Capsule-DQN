import tensorflow as tf

flags = tf.app.flags

############################
#    hyper parameters      #
############################

flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("height", 84, "frame height")
flags.DEFINE_integer("width", 84, "frame width")
flags.DEFINE_integer("state_length", 4, "length of state")
flags.DEFINE_integer("action_repeat", 4, "number of skip frame")
flags.DEFINE_integer("replay_start_size", 10000, "replay start size")
flags.DEFINE_integer("decay", 500000, "epsilon decay")
flags.DEFINE_float("min_epsilon", 0.1, "minimum epsilon")
flags.DEFINE_float("gamma", 0.99, "discount rate")
flags.DEFINE_integer("episode", 500, "episode length")
flags.DEFINE_integer("train_freq", 4, "frequence of updating network")
flags.DEFINE_integer("sync_freq", 10000, "frequence of target nets update")
flags.DEFINE_integer("eval_freq", 10, "frequence of evaluation")
flags.DEFINE_integer("N", 50000, "N")
flags.DEFINE_integer("routing_iters", 3, "number of iterations in routing algorithm")

############################
#   environment setting    #
############################

flags.DEFINE_string("env", "Pong-v0", "OpenAI Gym environment name")
flags.DEFINE_boolean("is_training", True, "training mode")
flags.DEFINE_boolean("render", True, "render")
flags.DEFINE_boolean("save", True, "save learned model")
flags.DEFINE_integer("save_freq", 10000, "frequence of saving params")
flags.DEFINE_boolean("restore", True, "restore model")
flags.DEFINE_string("stored_path", "./checkpoint/", "path for stored model")
flags.DEFINE_string("log_dir", "./summary/", "path for summary file")

cfg = flags.FLAGS
