import tensorflow as tf
import glob
import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from matplotlib import rc
import numpy as np
import scipy.stats as sp
from cycler import cycler

plt.rcParams["font.family"] = "Times New Roman" 
plt.rcParams['axes.prop_cycle'] = cycler('color',['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00'])
lite = False

def read_data(tf_dir, tag="return_mean", MAX_HORIZON=1000000000):

    list_of_data = []

    tf_files = glob.glob(tf_dir + "events.out.tfevents*")

    for tf_file in tf_files:
        done = False
        data = []
        frames = []

        for e in tf.compat.v1.train.summary_iterator(tf_file):
            for v in e.summary.value:
                if v.tag == "frames":
                    frames.append(v.simple_value)
                    if v.simple_value > MAX_HORIZON:
                        done = True
                elif v.tag == tag:
                    data.append(v.simple_value)
            if done:
                break

        data = data[:min(len(frames), len(data))]
        frames = frames[:min(len(frames), len(data))]

        if len(data) != 0:
            list_of_data.append((data, frames))

    list_of_data = sorted(list_of_data, key=(lambda pair: min(pair[1])))

    cleaned_frames = []
    cleaned_data = []
    max_frames = 0
    for (data, frames) in list_of_data:
        for i in range(len(data)):
            if frames[i] < max_frames:
                continue
            else:
                max_frames = frames[i]
                cleaned_frames.append(frames[i])
                cleaned_data.append(data[i])


    # Cumulative sum smoothing
    window_width = 1
    cumsum_vec = np.cumsum(np.insert(cleaned_data, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return cleaned_frames[:len(ma_vec)], ma_vec

# Returns (X_data, (Y_average, Y_error))
def read_data_and_average(tf_dirs, tag="return_mean", MAX_HORIZON=1000000000):
    X_all = None
    Ys = []


    if lite:
        tf_dirs = tf_dirs[:1]
    for tf_dir in tf_dirs:
        X, Y = read_data(tf_dir, tag, MAX_HORIZON)

        print(len(X))

        if len(X) < 60:
            continue

        Ys.append(Y)
        if X_all is None:
            X_all = X
        else:
            if len(X) < len(X_all):
                X_all = X

    N = min([len(Ys[i]) for i in range(len(Ys))])
    M = len(Ys)
    print(M, N, [len(Ys[i]) for i in range(len(Ys))])
    

    avgs = [0 for i in range(N)]
    lower = [0 for i in range(N)]
    upper = [0 for i in range(N)]

    for n in range(N):
        vals = []
        for m in range(M):
            vals.append(Ys[m][n])

        vals = np.array(vals)
        if M == 1:
            lo = hi = np.mean(vals)
        else:
            lo, hi = sp.t.interval(0.9, len(vals)-1, loc=np.mean(vals), scale=sp.sem(vals))

        avgs[n] = (lo+hi)/2
        lower[n] = lo
        upper[n] = hi

    return X_all, avgs, lower, upper

def plot_minigrid_env_dfa_experiments():

    logs = [glob.glob("storage/RGCN_8x32_ROOT_SHARED_Adversarial*_use-dfa:False_*/train/"), 
            glob.glob("storage/RGCN_8x32_ROOT_SHARED_Adversarial*_use-dfa:True_*/train/")]
            

    labels = ["AST RGCN (Baseline)", "DFA RGCN"]
    # labels = ["Deterministic Finite Automata", "Abstract Syntax Tree (Baseline)"]
    horizon = 10000000000

    fig, ax1 = plt.subplots(1, 1)
    plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0.12, left=0.07, right = 0.96)
    fig.set_size_inches(10,5)
    

    ax1.set_ylabel("Average total reward", fontsize = 16)
    ax1.tick_params(labelsize=12)
    ax1.set_title("Minigrid Environment", fontsize = 16)
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    for data_line, label in zip(logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="return_mean", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]
        ax1.plot(X, Y_avg, linewidth = 2, label=label)
        ax1.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    fig.text(0.5, 0.2, 'Frames (millions)', ha='center', fontsize = 16)
    plt.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc="lower right", markerscale=6, fontsize=16, ncol = 1)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/minigrid-env-dfa.pdf", bbox_inches='tight', pad_inches=0)
    # plt.show()

def plot_letter_env_new_sampling_experiments():

    # logs = [ glob.glob("fresh_storage/RGCN_8x32_ROOT_Until_1_3_1_2*dfa:True*/train/")]
    # labels = ["DFA RGCN"]

    logs = [glob.glob("fresh_storage/RGCN_8x32_ROOT_Until_1_2_1_1*dfa:True*/train/"),
            glob.glob("fresh_storage/RGCN_8x32_ROOT_SHARED_Until_1_2_1_1*dfa:True*/train/"),
            glob.glob("../../original_ltl2action/LTL2Action/src/storage/RGCN_8x32_ROOT_SHARED_Until_1_2_1_1*/train/") ]

    labels = ["DFA Root", "DFA Shared", "DFA Old", "AST Shared"]
    horizon = 10000000000

    fig, ax1 = plt.subplots(1, 1)
    plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0.12, left=0.07, right = 0.96)
    fig.set_size_inches(10,5)

    ax1.set_ylabel("Discounted return", fontsize = 16)
    ax1.tick_params(labelsize=12)
    ax1.set_title("Avoidance Tasks", fontsize = 16)
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))-Ordered 

    for data_line, label in zip(logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]
        ax1.plot(X, Y_avg, linewidth = 2, label=label)
        ax1.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    fig.text(0.5, 0.2, 'Frames (millions)', ha='center', fontsize = 16)
    plt.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc="lower right", markerscale=6, fontsize=16, ncol = 1)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/zone-until_1_2_1_1.pdf",bbox_inches='tight', pad_inches=0)


def plot_letter_env_comparison_experiments():

    eventually_logs = [glob.glob("storage/RGCN_8x32_ROOT_SHARED_Eventually*dfa:False*/train/"), 
                       glob.glob("storage/RGCN_8x32_ROOT_SHARED_Eventually*dfa:True*/train/"),
                       glob.glob("storage/GCN_8x32_ROOT_SHARED_Eventually*dfa:True*/train/")]

    labels = ["AST RGCN (Baseline)", "DFA RGCN", "DFA GCN"]
    # labels = ["Deterministic Finite Automata", "Abstract Syntax Tree (Baseline)"]
    horizon = 10000000000

    fig, ax1 = plt.subplots(1, 1)
    plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0.12, left=0.07, right = 0.96)
    fig.set_size_inches(10,5)

    ax1.set_ylabel("Discounted return", fontsize = 16)
    ax1.tick_params(labelsize=12)
    ax1.set_title("Partially-Ordered Tasks", fontsize = 16)
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))-Ordered 

    for data_line, label in zip(eventually_logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]
        ax1.plot(X, Y_avg, linewidth = 2, label=label)
        ax1.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    fig.text(0.5, 0.2, 'Frames (millions)', ha='center', fontsize = 16)
    plt.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc="lower right", markerscale=6, fontsize=16, ncol = 1)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/letter-env-comparison.pdf",bbox_inches='tight', pad_inches=0)

def plot_zone_env_until_comparison_experiments():

    eventually_logs = [glob.glob("storage/RGCN_8x32_ROOT_Until_1_2_1_1_Zones*dfa:True*use_onehot_guard_embed:False/train/")]

    # labels = ["AST RGCN (Baseline)", "DFA RGCN bugged", "DFA RGCN One-hot", "DFA GCN bugged", "DFA RGCN Non-shared One-hot", "DFA RGCN", "DFA RGCN Updated One-hot", "DFA RGCN Updated One-hot Non-shared"]
    # labels = ["RGCN AST (baseline)", "RGCN DFA", "RGCN One-hot", "RGCN MDP state"]
    labels = ["RGCN One-hot"]
    # labels = ["Deterministic Finite Automata", "Abstract Syntax Tree (Baseline)"]
    horizon = 10000000000

    fig, ax1 = plt.subplots(1, 1)
    plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0.12, left=0.07, right = 0.96)
    fig.set_size_inches(15,8)
    

    ax1.set_ylabel("Discounted return", fontsize = 16)
    ax1.tick_params(labelsize=12)
    ax1.set_title("Avoidance Tasks", fontsize = 16)
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    for data_line, label in zip(eventually_logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]
        ax1.plot(X, Y_avg, linewidth = 2, label=label)
        ax1.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    fig.text(0.5, 0.2, 'Frames (millions)', ha='center', fontsize = 16)
    plt.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc="lower right", markerscale=6, fontsize=16, ncol = 1)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/zone-env-until-comparison.pdf",bbox_inches='tight', pad_inches=0)



def plot_letter_env_until_comparison_experiments():

    # eventually_logs = [glob.glob("storage/RGCN_8x32_ROOT_SHARED_Until*dfa:False*/train/"),
    #                 glob.glob("storage/RGCN_8x32_ROOT_SHARED_Until_1_2_1_2*dfa:True*use_mean_guard_embed:False/train/"),
    #                    glob.glob("storage/RGCN_8x32_ROOT_Until*dfa:True*use_onehot_guard_embed:True/train/") ,
    #                    glob.glob("mdp_state_storage/RGCN_8x32_ROOT_Until*dfa:True*/train/") ]

    eventually_logs = [glob.glob("fresh_storage/RGCN_8x32_ROOT_Until_1_3_1_2*dfa:True*/train/")]

    # labels = ["AST RGCN (Baseline)", "DFA RGCN bugged", "DFA RGCN One-hot", "DFA GCN bugged", "DFA RGCN Non-shared One-hot", "DFA RGCN", "DFA RGCN Updated One-hot", "DFA RGCN Updated One-hot Non-shared"]
    labels = ["RGCN ROOT DFA"]
    # labels = ["Deterministic Finite Automata", "Abstract Syntax Tree (Baseline)"]
    horizon = 10000000000

    fig, ax1 = plt.subplots(1, 1)
    plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0.12, left=0.07, right = 0.96)
    fig.set_size_inches(15,8)
    

    ax1.set_ylabel("Discounted return", fontsize = 16)
    ax1.tick_params(labelsize=12)
    ax1.set_title("Avoidance Tasks", fontsize = 16)
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    for data_line, label in zip(eventually_logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]
        ax1.plot(X, Y_avg, linewidth = 2, label=label)
        ax1.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    fig.text(0.5, 0.2, 'Frames (millions)', ha='center', fontsize = 16)
    plt.grid(True)

    handles, labels = ax1.get_legend_handles_labels()
    legend = plt.legend(handles, labels, loc="lower right", markerscale=6, fontsize=16, ncol = 1)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/letter-env-until.pdf",bbox_inches='tight', pad_inches=0)


def plot_letter_env_dfa_experiments():

    eventually_logs = [glob.glob("storage/RGCN_8x32_ROOT_SHARED_Eventually*dfa:False*/train/"), 
            glob.glob("storage/RGCN_8x32_ROOT_SHARED_Eventually*dfa:True*use_mean_guard_embed:False/train/"),
                       glob.glob("storage/RGCN_8x32_ROOT_Eventually*dfa:True*use_onehot_guard_embed:True/train/"),
                       glob.glob("mdp_state_storage/RGCN_8x32_ROOT_Eventually*dfa:True*/train/") ]

    # labels = ["RGCN", "GCN"]
    labels = ["RGCN AST (baseline)", "RGCN DFA", "RGCN One-hot", "RGCN MDP state"]
    horizon = 10000000000

    fig, ax1 = plt.subplots(1, 1)
    plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0.12, left=0.07, right = 0.96)
    fig.set_size_inches(20,10)
    
    ax1.grid()

    ax1.set_xlim(0,2)

    ax1.set_ylabel("Discounted return", fontsize = 16)
    ax1.tick_params(labelsize=12)
    # ax1.set_title("Partially-Ordered Tasks", fontsize = 16)
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    for data_line, label in zip(eventually_logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]
        ax1.plot(X, Y_avg, linewidth = 2, label=label)
        ax1.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    fig.text(0.5, 0.2, 'Frames (millions)', ha='center', fontsize = 16)

    handles, labels = ax1.get_legend_handles_labels()
    legend = ax1.legend(handles, labels, loc="lower right", markerscale=6, fontsize=16, ncol = 1)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    #########################################################################
    # eventually_logs = [glob.glob("storage/RGCN_8x32_ROOT_SHARED_Eventually*dfa:True*/train/"),
    #                    glob.glob("storage/RGCN_8x32_ROOT_SHARED_Eventually*dfa:False*/train/")]

    # labels = ["Deterministic Finite Automata", "Abstract Syntax Tree"]
    # horizon = 10000000000

    # ax2.grid()
    # # ax2.set_ylabel("Discounted return", fontsize = 16)
    # ax2.tick_params(labelsize=12)
    # # ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # for data_line, label in zip(eventually_logs, labels):
    #     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
    #     X = [x / 1000000 for x in X]
    #     ax2.plot(X, Y_avg, linewidth = 2, label=label)
    #     ax2.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    #########################################################################


    plt.suptitle("Partially-Ordered Tasks", fontsize = 16)


    handles, labels = ax1.get_legend_handles_labels()
    legend = ax1.legend(handles, labels, loc="lower right", markerscale=6, fontsize=16, ncol = 1)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/letter-eventually-onehot-comparison.pdf",bbox_inches='tight', pad_inches=0)
    # plt.show()
    
def plot_letter_env_until_comparison():

    until_logs = [glob.glob("storage/RGCN_8x32_ROOT_SHARED_Until_1_2_1_2_Letter*/train/"),
                       glob.glob("storage/baseline_exps/RGCN_8x32_ROOT_SHARED_Until_1_2_1_2_Letter*/train/")]

    labels = ["DFA", "AST"]
    horizon = 1000000000

    fig, ax1 = plt.subplots(1, 1)
    plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0.12, left=0.07, right = 0.96)
    fig.set_size_inches(10,5)
    

    ax1.set_ylabel("Discounted return", fontsize = 16)
    ax1.tick_params(labelsize=12)
    ax1.set_title("Avoidance Tasks with RGCN_8x32_ROOT_SHARED", fontsize = 16)
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    for data_line, label in zip(until_logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]
        ax1.plot(X, Y_avg, linewidth = 2, label=label)
        ax1.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    fig.text(0.5, 0.2, 'Frames (millions)', ha='center', fontsize = 16)

    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center",bbox_to_anchor = (0.5, 0), markerscale=6, fontsize=16, ncol = 3)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/letter-env-until-comparison.pdf")
    # plt.show()


def plot_letter_env_experiments():
    until_logs = [glob.glob("storage-good/RGCN/*Until*/train/"),
                    glob.glob("storage-good/GRU/*Until*/train/"),
                    glob.glob("storage-good/LSTM/*Until*/train/"),
                    glob.glob("storage-good/Myopic/*Until*/train/"),
                    glob.glob("storage-good/GRU-no-progression/*Until*/train/"),
                    glob.glob("storage-good/No-LTL/*Until*/train/")
                ]

    eventually_logs = [glob.glob("storage-good/RGCN/*Eventually*/train/"),
                    glob.glob("storage-good/GRU/*Eventually*/train/"),
                    glob.glob("storage-good/LSTM/*Eventually*/train/"),
                    glob.glob("storage-good/Myopic/*Eventually*/train/"),
                    glob.glob("storage-good/GRU-no-progression/*Eventually*/train/"),
                    glob.glob("storage-good/No-LTL/*Eventually*/train/")
                ]

    labels = ["GNN+progression", "GRU+progression", "LSTM+progression", "Myopic", "GRU", "No LTL"]
    horizon = 20000000000

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0.12, left=0.07, right = 0.96)
    fig.set_size_inches(10,5)
    

    ax1.set_ylabel("Discounted return", fontsize = 16)
    ax1.tick_params(labelsize=12)
    ax1.set_title("Avoidance Tasks", fontsize = 16)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    for data_line, label in zip(until_logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]
        ax1.plot(X, Y_avg, linewidth = 2, label=label)
        ax1.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    ax2.tick_params(labelsize=12)
    ax2.set_title("Parallel Tasks", fontsize = 16)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    for data_line, label in zip(eventually_logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #return_mean, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]
        ax2.plot(X, Y_avg, linewidth = 2, label=label)
        ax2.fill_between(X, Y_lower, Y_upper, alpha=0.2)

    fig.text(0.5, 0.2, 'Frames (millions)', ha='center', fontsize = 16)

    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center",bbox_to_anchor = (0.5, 0), markerscale=6, fontsize=16, ncol = 3)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)


    plt.savefig("figs/letter-env.pdf")
    plt.show()



def plot_pretraining_experiments():
    until_logs = [glob.glob("storage-good/RGCN/*Until_1_3_1_2*/train/"),
                    glob.glob("transfer-storage/RGCN/*pretrained_Until_1_3_1_2*/train/"),
                    glob.glob("storage-good/GRU/*Until_1_3_1_2*/train/"),
                    glob.glob("transfer-storage/GRU/*pretrained_Until_1_3_1_2*/train/"),
                    glob.glob("storage-good/Myopic/*Until*/train/")
                ]


    eventually_logs = [glob.glob("storage-good/RGCN/*Eventually_1_5_1_4*/train/"),
                    glob.glob("transfer-storage/RGCN/*pretrained_Eventually_1_5_1_4*/train/"),
                    glob.glob("storage-good/GRU/*Eventually_1_5_1_4*/train/"),
                    glob.glob("transfer-storage/GRU/*pretrained_Eventually_1_5_1_4*/train/"),
                    glob.glob("storage-good/Myopic/*Eventually*/train/")
                ]

    labels = ["GNN+progression", "GNN+progression+pretraining", "GRU+progression", "GRU+progression+pretraining",   "Myopic"]
    horizon = 10000000

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(top = 0.92, bottom = 0.28, hspace = 0, wspace = 0.12, left=0.07, right = 0.96)
    fig.set_size_inches(10,5)
    

    ax1.set_ylabel("Discounted return", fontsize = 16)
    ax1.tick_params(labelsize=12)
    ax1.set_title("Avoidance Tasks", fontsize = 16)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))



    for data_line, label in zip(until_logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]

        if label == "GNN+progression":
            color = '#377eb8'
            style = "-"
        elif label == "GNN+progression+pretraining":
            color = '#377eb8'
            style = "dotted"
        elif label == "GRU+progression":
            color = "#ff7f00"
            style = "-"
        elif label == "GRU+progression+pretraining":
            color = "#ff7f00"
            style = "dotted"
        elif label == "Myopic":
            color = '#f781bf'
            style = "-"
        else:
            raise Exception("")

        ax1.plot(X, Y_avg, linewidth = 2, label=label, linestyle=style, color=color)
        ax1.fill_between(X, Y_lower, Y_upper, alpha=0.2, color=color)

    ax2.tick_params(labelsize=12)
    ax2.set_title("Parallel Tasks", fontsize = 16)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    for data_line, label in zip(eventually_logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return", MAX_HORIZON=horizon) #return_mean, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]

        if label == "GNN+progression":
            color = '#377eb8'
            style = "-"
        elif label == "GNN+progression+pretraining":
            color = '#377eb8'
            style = "dotted"
        elif label == "GRU+progression":
            color = "#ff7f00"
            style = "-"
        elif label == "GRU+progression+pretraining":
            color = "#ff7f00"
            style = "dotted"
        elif label == "Myopic":
            color = '#f781bf'
            style = "-"
        else:
            raise Exception("")

        ax2.plot(X, Y_avg, linewidth = 2, label=label, linestyle=style, color=color)
        ax2.fill_between(X, Y_lower, Y_upper, alpha=0.2, color=color)

    fig.text(0.5, 0.2, 'Frames (millions)', ha='center', fontsize = 16)

    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="lower center",bbox_to_anchor = (0.5, 0), markerscale=6, fontsize=16, ncol = 3)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/transfer.pdf")
    plt.show()


def plot_safety_experiments():



# plt.figure(figsize=(4,3))
# plt.ylabel("Discounted return")
# plt.xlabel("Frames")
# plt.title(title)

# for data_line, label in lines:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="return_mean") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label)
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2)




# plt.legend()
# # plt.savefig("figs/" + title + ".pdf")
# plt.show()


    A_logs = glob.glob("zone-good/pretrained/*/train/")
    B_logs = glob.glob("zone-good/RGCN/*/train/")
    C_logs = glob.glob("zone-good/myopic/*/train/")
    

    logs = [A_logs, B_logs, C_logs]
    labels = ["GNN+progression+pretraining", "GNN+progression", "Myopic"]

    horizon = 20000000

    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(7,5)
    

    plt.ylabel("Total reward", fontsize = 16)
    plt.tick_params(labelsize=12)
    plt.title("Avoidance Tasks", fontsize = 16)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    for data_line, label in zip(logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="return_mean", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]

        if label == "GNN+progression+pretraining":
            color = '#377eb8'
            style = "dotted"
        elif label == "Myopic":
            color = '#f781bf'
            style = "-"
        elif label == "GNN+progression":
            color = '#377eb8'
            style = "-"
        else:
            raise Exception("")

        plt.plot(X, Y_avg, linewidth = 2, label=label, color=color, linestyle=style)
        plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color=color)

    plt.xlabel('Frames (millions)', fontsize = 16)

    handles, labels = ax1.get_legend_handles_labels()
    legend = plt.legend(handles, labels, markerscale=6, fontsize=16)

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/safety.pdf")
    plt.show()


def plot_toy_experiments():
    # Toy

    A_logs = glob.glob("toy-minigrid/RGCN/*/train/")
    B_logs = glob.glob("toy-minigrid/myopic/*/train/")
    

    logs = [A_logs, B_logs]
    labels = ["GNN+progression", "Myopic"]
    horizon = 3000000

    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(7,5)
    

    plt.ylabel("Total reward", fontsize = 16)
    plt.tick_params(labelsize=12)
    #plt.title("Avoidance Tasks", fontsize = 16)
    #ax1.xaxis.set_major_formatter(ScalarFormatter())
    plt.ylim([0,1.05])

    for data_line, label in zip(logs, labels):
        X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="return_mean", MAX_HORIZON=horizon) #average_discounted_return, average_discounted_return, average_reward_per_step
        X = [x / 1000000 for x in X]

        if label == "GNN+progression+pretraining":
            color = '#377eb8'
            style = "dotted"
        elif label == "Myopic":
            color = '#f781bf'
            style = "-"
        elif label == "GNN+progression":
            color = '#377eb8'
            style = "-"
        else:
            raise Exception("")

        plt.plot(X, Y_avg, linewidth = 2, label=label, color=color, linestyle=style)
        plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color=color)

    plt.xlabel('Frames (millions)', fontsize = 16)  

    handles, labels = ax1.get_legend_handles_labels()
    legend = plt.legend(handles, labels, markerscale=6, fontsize=16, loc="lower right")

    for i in range(len(legend.get_lines())):
        legend.get_lines()[i].set_linewidth(3)

    plt.savefig("figs/toy.pdf")
    plt.show()

# # # Letter-Env UNTIL_1_3_1_2
# A_logs = glob.glob("storage-good/RGCN/*Until*/train/")
# B_logs = glob.glob("storage-good/GRU/*Until*/train/")
# C_logs = glob.glob("storage-good/LSTM/*Until*/train/")
# D_logs = glob.glob("storage-good/Myopic/*Until*/train/")
# E_logs = glob.glob("storage-good/GRU-no-progression/*Until*/train/")
# F_logs = glob.glob("storage-good/No-LTL/*Until*/train/")
# lines = [(A_logs, "GNN+progression"), (B_logs, "GRU+progression"), (C_logs, "LSTM+progression"), (D_logs, "Myopic"), (E_logs, "GRU"), (F_logs, "No LTL")]
# MAX_HORIZON = 20000000
# title = "Letter-Env: Until_1_3_1_2"


# # Letter-Env EVENTUALLY_1_5_1_4
# A_logs = glob.glob("storage-good/RGCN/*Eventually*/train/")
# B_logs = glob.glob("storage-good/GRU/*Eventually*/train/")
# C_logs = glob.glob("storage-good/LSTM/*Eventually*/train/")
# D_logs = glob.glob("storage-good/Myopic/*Eventually*/train/")
# E_logs = glob.glob("storage-good/GRU-no-progression/*Eventually*/train/")
# F_logs = glob.glob("storage-good/No-LTL/*Eventually*/train/")

# lines = [(A_logs, "GNN+progression"), (B_logs, "GRU+progression"), (C_logs, "LSTM+progression"), (D_logs, "Myopic"), (E_logs, "GRU"), (F_logs, "No LTL")]
# MAX_HORIZON = 20000000
# title = "Letter-Env: Eventually_1_5_1_4"

# #Symbol-Env UNTIL_1_3_1_2

# C_logs = glob.glob("symbol-storage/RGCN/*Until_1_3_1_2*/train/")
# D_logs = glob.glob("symbol-storage/GRU/*Until_1_3_1_2*/train/")

# lines = [(C_logs, "GNN"), (D_logs, "GRU")]
# title = "Symbol Env: Until_1_3_1_2"


# #Symbol-Env EVENTUALLY_1_5_1_4
# A_logs = glob.glob("symbol-storage/RGCN/*Eventually_1_5_1_4*/train/")
# B_logs = glob.glob("symbol-storage/GRU/*Eventually_1_5_1_4*epochs:4*lr:0.003*/train/")

# lines = [(A_logs, "GNN"), (B_logs, "GRU")]
# title = "Symbol Env: Eventually_1_5_1_4"

#=============================================================================================================================================
# ## NOTE: For the transfer results you need to comment out the code at the bottom. 

# # Transfer Until_1_3_1_2
# A1_logs = glob.glob("storage-good/RGCN/*Until_1_3_1_2*/train/")
# A2_logs = glob.glob("transfer-storage/RGCN/*pretrained_Until_1_3_1_2*/train/")
# A3_logs = glob.glob("transfer-storage/RGCN/*pretrained-freeze_ltl_Until_1_3_1_2*bs:64*/train/")

# B1_logs = glob.glob("storage-good/GRU/*Until_1_3_1_2*/train/")
# B2_logs = glob.glob("transfer-storage/GRU/*pretrained_Until_1_3_1_2*/train/")
# B3_logs = glob.glob("transfer-storage/GRU/*pretrained-freeze_ltl_Until_1_3_1_2*/train/")

# lines_A = [(A1_logs, "GNN", "-"), (A2_logs, "GNN+pretraining", "dotted")]
# lines_B = [(B1_logs, "GRU", "-"), (B2_logs, "GRU+pretraining", "dotted")]
# MAX_HORIZON = 10000000

# title = "Transfer - Letter-Env: Until_1_3_1_2"
# plt.ylabel("Discounted return")
# plt.xlabel("Frames")

# for data_line, label, linestyle in lines_A:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label, linestyle=linestyle, color="b")
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color="b")

# for data_line, label, linestyle in lines_B:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label, linestyle=linestyle, color="orange")
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color="orange")

# plt.legend()
# plt.show()

#=============================================================================================================================================


#  #Transfer EVENTUALLY_1_5_1_4
# A1_logs = glob.glob("storage-good/RGCN/*Eventually_1_5_1_4*/train/")
# A2_logs = glob.glob("transfer-storage/RGCN/*pretrained_Eventually_1_5_1_4*/train/")
# A3_logs = glob.glob("transfer-storage/RGCN/*pretrained-freeze_ltl_Eventually_1_5_1_4*bs:64*/train/")

# B1_logs = glob.glob("storage-good/GRU/*Eventually_1_5_1_4*/train/")
# B2_logs = glob.glob("transfer-storage/GRU/*pretrained_Eventually_1_5_1_4*/train/")
# B3_logs = glob.glob("transfer-storage/GRU/*pretrained-freeze_ltl_Eventually_1_5_1_4*/train/")

# lines_A = [(A1_logs, "GNN", "-"), (A2_logs, "GNN+pretraining", "dotted")]
# lines_B = [(B1_logs, "GRU", "-"), (B2_logs, "GRU+pretraining", "dotted")]
# MAX_HORIZON = 10000000
# title = "Transfer - Letter-Env: Eventually_1_5_1_4"
# plt.ylabel("Discounted return")
# plt.xlabel("Frames")

# for data_line, label, linestyle in lines_A:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label, linestyle=linestyle, color="b")
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color="b")

# for data_line, label, linestyle in lines_B:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="average_discounted_return") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label, linestyle=linestyle, color="orange")
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2, color="orange")

# plt.legend()
# plt.show()

#=============================================================================================================================================


# #Safety
# A_logs = glob.glob("zone-good/pretrained/*/train/")
# B_logs = glob.glob("zone-good/myopic/*/train/")

# lines = [(A_logs, "Best GNN so far"),(B_logs, "Best Myopic")]
# title = "Safety"

# #=========================================

# # Toy

# # A_logs = glob.glob("toy-minigrid/RGCN/*/train/")
# # B_logs = glob.glob("toy-minigrid/myopic/*/train/")
# # lines = [(A_logs, "GNN"), (B_logs, "Myopic")]
# # title = "Toy Minigrid"


# plt.figure(figsize=(4,3))
# plt.ylabel("Discounted return")
# plt.xlabel("Frames")
# plt.title(title)

# for data_line, label in lines:
#     X, Y_avg, Y_lower, Y_upper = read_data_and_average(data_line, tag="return_mean") #return_mean, average_discounted_return, average_reward_per_step
#     plt.plot(X, Y_avg, linewidth = 1, label=label)
#     plt.fill_between(X, Y_lower, Y_upper, alpha=0.2)




# plt.legend()
# # plt.savefig("figs/" + title + ".pdf")
# plt.show()


#plot_letter_env_experiments()
# plot_letter_env_dfa_experiments()
plot_letter_env_new_sampling_experiments()
# plot_letter_env_until_comparison_experiments()
# plot_letter_env_until_comparison_experiments()
# plot_minigrid_env_dfa_experiments()
# plot_pretraining_experiments()
# plot_safety_experiments()
# plot_toy_experiments()
