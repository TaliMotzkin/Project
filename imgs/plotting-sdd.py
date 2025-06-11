import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

epochs = np.array([2, 4, 8, 10, 14, 30])

models = ['Base-Rigid', 'Base-Smooth', '10GAN', '100GAN']

plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 13
})
#################NBA#########################
#####GROUPNET#################
# Data for Baseline
base_dus = np.array([0.52, 0.5, 0.51, 0.53, 0.52, 0.53])
base_dus_std = np.array([0.03, 0.03, 0.03, 0.04,0.05, 0.05])

base_dcs =  np.array([0.43, 0.37, 0.38, 0.35, 0.33, 0.31])
base_dcs_std = np.array([0.06, 0.08, 0.09,0.10, 0.11, 0.12])

base_recall =  np.array([0.89, 0.93, 0.92, 0.91, 0.93, 0.92])
base_recall_std = np.array([0.17,0.11, 0.15,0.16, 0.14,0.15, ])

base_acc = np.array([0.62, 0.54, 0.57, 0.62, 0.64, 0.67])
base_acc_std =  np.array([0.17, 0.08, 0.11, 0.18, 0.15, 0.16])

base_per =  np.array([0.59, 0.54, 0.54, 0.6, 0.6, 0.62])
base_per_std =  np.array([0.13, 0.05, 0.07, 0.14, 0.12, 0.14])

base_f1 = np.array([0.7, 0.68, 0.69, 0.7, 0.72, 0.74])
base_f1_std =  np.array([0.13, 0.06, 0.07, 0.13, 0.11, 0.12])



# Data for Closest
closest_dus =  np.array([0.61, 0.72, 0.84, 0.85, 0.88, 0.94])
closest_dus_std = np.array([0.03, 0.05, 0.06, 0.06, 0.06, 0.06])

closest_dcs =  np.array([0.57, 0.28,0.21, 0.17, 0.15, 0.11])
closest_dcs_std = np.array([0.03, 0.20, 0.22, 0.20, 0.17, 0.13])

closest_recall =  np.array([0.42, 0.91, 0.93, 0.94, 0.95, 0.98])
closest_recall_std = np.array([0.27,0.16, 0.13,0.12, 0.12,0.05, ])
closest_acc = np.array([0.42, 0.66, 0.68, 0.68, 0.69, 0.7])
closest_acc_std =  np.array([0.28, 0.18, 0.17, 0.16, 0.16, 0.14])

closest_per =  np.array([0.42, 0.62, 0.63, 0.64, 0.64, 0.65])
closest_per_std =  np.array([0.27, 0.15, 0.14, 0.14, 0.13, 0.12])

closest_f1 = np.array([0.42, 0.73, 0.75, 0.75, 0.76, 0.77])
closest_f1_std =  np.array([0.27, 0.14, 0.13, 0.12, 0.12, 0.09])


gan10_dcs =  np.array([0.18, 0.11, 0.09, 0.07, 0.08, 0.05])
gan10_dcs_std = np.array([0.14,0.12, 0.09,0.08,0.08,0.06])

gan10_dus =  np.array([0.77, 0.8, 0.88, 0.89, 0.94, 0.94])
gan10_dus_std = np.array([0.06,0.07,0.07, 0.07,0.05,0.06])

gan10_recall = np.array([0.97, 0.99, 1, 0.99, 1, 1])
gan10_recall_std =np.array([0.05,0.02,0.0, 0.02,0.0,0])


gan10_acc = np.array([0.7, 0.7, 0.49, 0.57, 0.61, 0.62])
gan10_acc_std =  np.array([0.14, 0.14, 0.05, 0.20, 0.20, 0.21])

gan10_per =  np.array([0.64, 0.64, 0.49, 0.56, 0.6, 0.6])
gan10_per_std =  np.array([0.12, 0.12, 0.03, 0.17, 0.16, 0.17])

gan10_f1 = np.array([0.74, 0.78, 0.65, 0.64, 0.69, 0.69])
gan10_f1_std =  np.array([0.09, 0.09, 0.03, 0.16, 0.16, 0.17])




gan100_dcs = np.array([0.5, 0.58, 0.44, 0.38, 0.36, 0.22])
gan100_dcs_std = np.array([0.01,0.1,0.09,0.09,0.09,0.1])

gan100_dus = np.array([0.5, 0.69, 0.59, 0.63, 0.7, 0.85])
gan100_dus_std = np.array([0.01,0.05, 0.07,0.07,0.06,0.07])

gan100_recall =  np.array([0.64, 0.51, 0.84, 0.94, 0.96, 0.99])
gan100_recall_std = np.array([0.25,0.26,0.17, 0.1,0.07,0.03])
gan100_acc = np.array([0.5, 0.54, 0.65, 0.67, 0.65, 0.72])
gan100_acc_std =  np.array([0.05, 0.25, 0.19, 0.18, 0.2 , 0.14])

gan100_per =  np.array([0.5, 0.54, 0.62, 0.63, 0.62, 0.66])
gan100_per_std =  np.array([0.03, 0.23, 0.16, 0.14, 0.17, 0.13])

gan100_f1 = np.array([0.66, 0.58, 0.71, 0.74, 0.71, 0.79])
gan100_f1_std =  np.array([0.04, 0.23, 0.15, 0.13, 0.17, 0.09])

# Data for 10GM-no mission
gm10_nomission =  np.array([0.36, 0.26, 0.18, 0.12, 0.15, 0.05])
gm10_nomission_std = np.array([0.14, 0.12, 0.1, 0.06, 0.1, 0.04])

# Data for 100GM-no mission
gm100_nomission = np.array([0.5,0.49, 0.33, 0.28, 0.23, 0.1])
gm100_nomission_std = np.array([0.01, 0.12, 0.08, 0.08, 0.03, 0.05])



gan250_dcs =np.array([0.50,0.49, 0.52, 0.51, 0.51, 0.5])
gan250_dcs_std = np.array([0.01, 0.02,0.02, 0.02, 0.03, 0.03])

gan250_dus = np.array([0.5, 0.49, 0.53, 0.51, 0.53, 0.52])
gan250_dus_std = np.array([0.01, 0.02, 0.02,0.02, 0.02, 0.03])

gan250_recall = np.array([0.59, 0.82, 0.49, 0.61, 0.6, 0.61])
gan250_recall_std = np.array([0.26, 0.18, 0.25, 0.24, 0.24, 0.23])

gan250_acc = np.array([0.43, 0.49, 0.44, 0.46, 0.48, 0.47])
gan250_acc_std =  np.array([0.19, 0.13, 0.25, 0.22, 0.23, 0.21])

gan250_per = np.array([0.44, 0.49, 0.45, 0.47,0.49, 0.48])
gan250_per_std =  np.array([0.16, 0.09, 0.23, 0.19, 0.2, 0.18])

gan250_f1 = np.array([0.5, 0.61, 0.47, 0.53, 0.53, 0.53])
gan250_f1_std =  np.array([0.19, 0.11, 0.24 , 0.2, 0.21, 0.19])

gan250_nomission =np.array([0.5, 0.49, 0.53, 0.51, 0.51, 0.5])
gan250_nomission_std = np.array([0.01, 0.02, 0.02, 0.02, 0.01,  0.03])


fig, axs = plt.subplots(1, 3, figsize=(20, 5))


axs[0].plot(epochs, base_dcs, label='Base-Rigid')
axs[0].fill_between(epochs, base_dcs - base_dcs_std, base_dcs + base_dcs_std, alpha=0.2)
axs[0].plot(epochs, closest_dcs, label='Base-Smooth')
axs[0].fill_between(epochs, closest_dcs - closest_dcs_std, closest_dcs + closest_dcs_std, alpha=0.2)
axs[0].plot(epochs, gan10_dcs, label='10GAN' ,color='green')
axs[0].fill_between(epochs, gan10_dcs - gan10_dcs_std, gan10_dcs + gan10_dcs_std, alpha=0.2,color='green')
axs[0].plot(epochs, gan100_dcs, label='100GAN',color='red')
axs[0].fill_between(epochs, gan100_dcs - gan100_dcs_std, gan100_dcs + gan100_dcs_std, alpha=0.2,color='red')
axs[0].plot(epochs, gan250_dcs, label='250GAN',color='purple')
axs[0].fill_between(epochs, gan250_dcs - gan250_dcs_std, gan250_dcs + gan250_dcs_std, alpha=0.2,color='purple')
axs[0].set_title("(a) Discriminator's controlled Agents' Scores\nSDD dataset - GroupNet Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Score")
axs[0].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[0].grid(False)

# axs[1].plot(epochs, base_dus, label='Base-Rigid')
# axs[1].fill_between(epochs, base_dus - base_dus_std, base_dus + base_dus_std, alpha=0.2)
# axs[1].plot(epochs, closest_dus, label='Base-Smooth')
# axs[1].fill_between(epochs, closest_dus - closest_dus_std, closest_dus + closest_dus_std, alpha=0.2)
# axs[1].plot(epochs, gan10_dus, label='10GAN',color='green')
# axs[1].fill_between(epochs, gan10_dus - gan10_dus_std, gan10_dus + gan10_dus_std, alpha=0.2,color='green')
# axs[1].plot(epochs, gan100_dus, label='100GAN',color='red')
# axs[1].fill_between(epochs, gan100_dus - gan100_dus_std, gan100_dus + gan100_dus_std, alpha=0.2,color='red')
# axs[1].plot(epochs, gan250_dus, label='250GAN',color='purple')
# axs[1].fill_between(epochs, gan250_dus - gan250_dus_std, gan250_dus + gan250_dus_std, alpha=0.2,color='purple')
# axs[1].set_title("Discriminator's uncntrolled Agents' Scores (DUS)\nSDD dataset - GroupNet Scheme", wrap=True)
# axs[1].set_xlabel("Discriminator's epoch")
# axs[1].set_ylabel("Score")
# axs[1].legend(loc='lower right', bbox_to_anchor=(0.99, 0.2))
# axs[1].grid(False)

axs[1].plot(epochs, gm10_nomission, label='10GM-no mission',  color='green')
axs[1].fill_between(epochs, gm10_nomission - gm10_nomission_std, gm10_nomission + gm10_nomission_std, alpha=0.2,  color='green')

axs[1].plot(epochs, gm100_nomission, label='100GM-no mission',  color='red')
axs[1].fill_between(epochs, gm100_nomission - gm100_nomission_std, gm100_nomission + gm100_nomission_std, alpha=0.2,  color='red')

axs[1].plot(epochs, gan250_nomission, label='250GM-no mission',  color='purple')
axs[1].fill_between(epochs, gan250_nomission - gan250_nomission_std, gan250_nomission +gan250_nomission_std, alpha=0.2,  color='purple')

axs[1].set_title("(b) Controlled (No Mission) Agents' Scores\nSDD dataset - GroupNet Scheme")
axs[1].set_xlabel("Discriminator's epoch")
axs[1].set_ylabel("Score")
axs[1].legend()
axs[1].grid(False)

axs[2].plot(epochs, base_recall, label='Base-Rigid')
axs[2].fill_between(epochs, base_recall - base_recall_std, base_recall + base_recall_std, alpha=0.2)
axs[2].plot(epochs, closest_recall, label='Base-Smooth')
axs[2].fill_between(epochs, closest_recall - closest_recall_std, closest_recall + closest_recall_std, alpha=0.2)
axs[2].plot(epochs, gan10_recall, label='10GAN',color='green')
axs[2].fill_between(epochs, gan10_recall - gan10_recall_std, gan10_recall + gan10_recall_std, alpha=0.2,color='green')
axs[2].plot(epochs, gan100_recall, label='100GAN',color='red')
axs[2].fill_between(epochs, gan100_recall - gan100_recall_std, gan100_recall +gan100_recall_std, alpha=0.2,color='red')
axs[2].plot(epochs, gan250_recall, label='250GAN',color='purple')
axs[2].fill_between(epochs, gan250_recall - gan250_recall_std, gan250_recall + gan250_recall_std, alpha=0.2,color='purple')

axs[2].set_title("(c) Agents' Recall\nSDD dataset - GroupNet Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("Recall")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(1, 1, figsize=(8, 5))

axs.plot(epochs, base_dus, label='Base-Rigid')
axs.fill_between(epochs, base_dus - base_dus_std, base_dus + base_dus_std, alpha=0.2)
axs.plot(epochs, closest_dus, label='Base-Smooth')
axs.fill_between(epochs, closest_dus - closest_dus_std, closest_dus + closest_dus_std, alpha=0.2)
axs.plot(epochs, gan10_dus, label='10GAN',color='green')
axs.fill_between(epochs, gan10_dus - gan10_dus_std, gan10_dus + gan10_dus_std, alpha=0.2,color='green')
axs.plot(epochs, gan100_dus, label='100GAN',color='red')
axs.fill_between(epochs, gan100_dus - gan100_dus_std, gan100_dus + gan100_dus_std, alpha=0.2,color='red')
axs.plot(epochs, gan250_dus, label='250GAN',color='purple')
axs.fill_between(epochs, gan250_dus - gan250_dus_std, gan250_dus + gan250_dus_std, alpha=0.2,color='purple')
axs.set_title("Discriminator's uncontrolled Agents' Scores\nSDD dataset - GroupNet Scheme", wrap=True)
axs.set_xlabel("Discriminator's epoch")
axs.set_ylabel("Score")
axs.legend(loc='lower right', bbox_to_anchor=(0.99, 0.2))
axs.grid(False)

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].plot(epochs, base_acc, label='Base-Rigid')
axs[0].fill_between(epochs, base_acc - base_acc_std, base_acc + base_acc_std, alpha=0.2)
axs[0].plot(epochs, closest_acc, label='Base-Smooth')
axs[0].fill_between(epochs, closest_acc - closest_acc_std, closest_acc + closest_acc_std, alpha=0.2)
axs[0].plot(epochs, gan10_acc, label='10GAN' ,color='green')
axs[0].fill_between(epochs, gan10_acc - gan10_acc_std, gan10_acc + gan10_acc_std, alpha=0.2,color='green')
axs[0].plot(epochs, gan100_acc, label='100GAN',color='red')
axs[0].fill_between(epochs, gan100_acc - gan100_acc_std, gan100_acc + gan100_acc_std, alpha=0.2,color='red')
axs[0].plot(epochs, gan250_acc, label='250GAN',color='purple')
axs[0].fill_between(epochs, gan250_acc - gan250_acc_std, gan250_acc + gan250_acc_std, alpha=0.2,color='purple')
axs[0].set_title("Agents' Accuracy\nSDD dataset - GroupNet Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Accuracy")
axs[0].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[0].grid(False)

axs[1].plot(epochs, base_per, label='Base-Rigid')
axs[1].fill_between(epochs, base_per - base_per_std, base_per + base_per_std, alpha=0.2)
axs[1].plot(epochs, closest_per, label='Base-Smooth')
axs[1].fill_between(epochs, closest_per - closest_per_std, closest_per + closest_per_std, alpha=0.2)
axs[1].plot(epochs, gan10_per, label='10GAN',color='green')
axs[1].fill_between(epochs, gan10_per - gan10_per_std, gan10_per + gan10_per_std, alpha=0.2,color='green')
axs[1].plot(epochs, gan100_per, label='100GAN',color='red')
axs[1].fill_between(epochs, gan100_per - gan100_per_std, gan100_per + gan100_per_std, alpha=0.2,color='red')
axs[1].plot(epochs, gan250_per, label='250GAN',color='purple')
axs[1].fill_between(epochs, gan250_per - gan250_per_std, gan250_per + gan250_per_std, alpha=0.2,color='purple')
axs[1].set_title("Agents' Precision\nSDD dataset - GroupNet Scheme", wrap=True)
axs[1].set_xlabel("Discriminator's epoch")
axs[1].set_ylabel("Precision")
axs[1].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[1].grid(False)

axs[2].plot(epochs, base_f1, label='Base-Rigid')
axs[2].fill_between(epochs, base_f1 - base_f1_std, base_f1 + base_f1_std, alpha=0.2)
axs[2].plot(epochs, closest_f1, label='Base-Smooth')
axs[2].fill_between(epochs, closest_f1 - closest_f1_std, closest_f1 +closest_f1_std, alpha=0.2)
axs[2].plot(epochs, gan10_f1, label='10GAN',color='green')
axs[2].fill_between(epochs, gan10_f1 - gan10_f1_std, gan10_f1 + gan10_f1_std, alpha=0.2,color='green')
axs[2].plot(epochs, gan100_f1, label='100GAN',color='red')
axs[2].fill_between(epochs, gan100_f1 - gan100_f1_std, gan100_f1 +gan100_f1_std, alpha=0.2,color='red')
axs[2].plot(epochs, gan250_f1, label='250GAN',color='purple')
axs[2].fill_between(epochs, gan250_f1 - gan250_f1_std, gan250_f1 + gan250_f1_std, alpha=0.2,color='purple')

axs[2].set_title("Agents' F1 Score\nSDD dataset - GroupNet Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("F1")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()






#####SAMPLER#################
# Data for Baseline
base_dus =  np.array([0.52, 0.56, 0.6, 0.54, 0.57, 0.59])
base_dus_std = np.array([0.02, 0.04, 0.06, 0.04,0.05, 0.06])

base_dcs =  np.array([0.46,0.4,0.37,0.32, 0.32, 0.25])
base_dcs_std = np.array([0.03, 0.08, 0.11,0.1, 0.12, 0.13])



base_acc = np.array([0.64, 0.66, 0.68, 0.67, 0.68, 0.69])
base_acc_std =  np.array([0.17, 0.16, 0.17, 0.15, 0.16 , 0.15])

base_per =  np.array([0.61, 0.62, 0.64, 0.62, 0.63, 0.62])
base_per_std =  np.array([0.14 ,0.14, 0.15, 0.12, 0.13, 0.12])

base_recall = np.array([0.89, 0.92, 0.92, 0.95, 0.93, 0.96])
base_recall_std = np.array([0.15 , 0.15, 0.15, 0.11, 0.12, 0.08])

base_f1 = np.array([0.71, 0.73, 0.74, 0.75, 0.73, 0.76])
base_f1_std =  np.array([0.13, 0.13, 0.14, 0.11, 0.12, 0.1])




# Data for Closest
closest_dus = np.array([0.62, 0.59, 0.59, 0.63, 0.84, 0.86])
closest_dus_std = np.array([0.03, 0.03, 0.05, 0.11, 0.06, 0.07])

closest_dcs =  np.array([0.61, 0.56, 0.42, 0.25, 0.2, 0.13])
closest_dcs_std = np.array([0.01, 0.02, 0.08, 0.16, 0.2, 0.17])



closest_acc = np.array([0.46, 0.47, 0.68, 0.67, 0.7, 0.71])
closest_acc_std =  np.array([0.28, 0.28, 0.18, 0.14, 0.16, 0.15])

closest_per =  np.array([0.46, 0.47 , 0.64, 0.63, 0.66, 0.66])
closest_per_std =  np.array([0.28, 0.28, 0.15, 0.12, 0.13, 0.13])

closest_recall = np.array([0.46, 0.48, 0.91, 0.95, 0.95, 0.96])
closest_recall_std = np.array([0.28, 0.27, 0.14, 0.08, 0.09, 0.07])

closest_f1 = np.array([0.46, 0.48, 0.75, 0.75, 0.77, 0.78])
closest_f1_std =  np.array([0.28, 0.28, 0.13, 0.09, 0.11, 0.1])


gan10_dcs_S =  np.array([0.44, 0.47, 0.48, 0.49, 0.43, 0.48])
gan10_dcs_std_S = np.array([0.04, 0.05, 0.07, 0.06, 0.06, 0.07])

gan10_dus_S =  np.array([0.51, 0.52, 0.53, 0.53, 0.51, 0.55])
gan10_dus_std_S = np.array([0.07,0.07, 0.07, 0.06, 0.09, 0.08])



gan10_acc = np.array([0.57, 0.55, 0.54, 0.53, 0.57, 0.56])
gan10_acc_std =  np.array([0.14, 0.15, 0.19, 0.21, 0.15, 0.21])

gan10_per =  np.array([0.55, 0.54, 0.53, 0.52, 0.55, 0.55])
gan10_per_std =  np.array([0.11, 0.12, 0.15, 0.18, 0.11, 0.17])

gan10_recall_S =  np.array([0.9, 0.85, 0.77, 0.71, 0.91, 0.76])
gan10_recall_std_S =np.array([0.14, 0.17, 0.22 , 0.23, 0.14, 0.23])

gan10_f1 = np.array([0.68, 0.65, 0.62, 0.6, 0.68, 0.63])
gan10_f1_std =  np.array([0.11, 0.12, 0.17, 0.19, 0.11, 0.18])



gan100_dcs_S =  np.array([0.48, 0.49, 0.47, 0.45, 0.46, 0.44])
gan100_dcs_std_S = np.array([0.01, 0.03, 0.03, 0.03, 0.03, 0.04])

gan100_dus_S =  np.array([0.51, 0.53, 0.51, 0.5, 0.5, 0.52])
gan100_dus_std_S = np.array([0.03,0.04, 0.03, 0.03, 0.03, 0.05])



gan100_acc = np.array([0.59, 0.57,0.61, 0.58 , 0.59, 0.62])
gan100_acc_std =  np.array([0.16, 0.2 ,0.14, 0.11, 0.12, 0.12])

gan100_per =  np.array([0.57, 0.55, 0.58, 0.55, 0.56, 0.58])
gan100_per_std =  np.array([0.12, 0.16, 0.11, 0.07, 0.09, 0.09])

gan100_recall_S =  np.array([0.88, 0.79, 0.9, 0.95, 0.93, 0.94])
gan100_recall_std_S = np.array([0.15, 0.21, 0.11, 0.09, 0.09, 0.08])

gan100_f1 = np.array([0.68, 0.75, 0.7, 0.65, 0.7, 0.74])
gan100_f1_std =  np.array([0.13, 0.17, 0.1, 0.07, 0.08, 0.08])

# Data for 10GM-no mission
gm10_nomission_S = np.array([0.44, 0.46, 0.47, 0.48, 0.43, 0.47])
gm10_nomission_std_S = np.array([0.04, 0.05, 0.07,  0.06 , 0.06, 0.07])

# Data for 100GM-no mission
gm100_nomission_S =  np.array([0.49, 0.48, 0.48, 0.47, 0.47, 0.46])
gm100_nomission_std_S = np.array([0.02, 0.03, 0.03, 0.03, 0.03, 0.04])





fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].plot(epochs, base_dcs, label='Base-Rigid')
axs[0].fill_between(epochs, base_dcs - base_dcs_std, base_dcs + base_dcs_std, alpha=0.2)
axs[0].plot(epochs, closest_dcs, label='Base-Smooth')
axs[0].fill_between(epochs, closest_dcs - closest_dcs_std, closest_dcs + closest_dcs_std, alpha=0.2)
axs[0].plot(epochs, gan10_dcs_S, label='10GAN',color='green')
axs[0].fill_between(epochs, gan10_dcs_S - gan10_dcs_std_S, gan10_dcs_S + gan10_dcs_std_S, alpha=0.2,color='green')
axs[0].plot(epochs, gan100_dcs_S, label='100GAN',color='red')
axs[0].fill_between(epochs, gan100_dcs_S - gan100_dcs_std_S, gan100_dcs_S + gan100_dcs_std_S, alpha=0.2,color='red')
axs[0].set_title("(a) Discriminator's controlled Agents' Scores\nSDD dataset - Sampler Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Score")
axs[0].legend()
axs[0].grid(False)

# axs[1].plot(epochs, base_dus, label='Base-Rigid')
# axs[1].fill_between(epochs, base_dus - base_dus_std, base_dus + base_dus_std, alpha=0.2)
# axs[1].plot(epochs, closest_dus, label='Base-Smooth')
# axs[1].fill_between(epochs, closest_dus - closest_dus_std, closest_dus + closest_dus_std, alpha=0.2)
# axs[1].plot(epochs, gan10_dus_S, label='10GAN',color='green')
# axs[1].fill_between(epochs, gan10_dus_S - gan10_dus_std_S, gan10_dus_S + gan10_dus_std_S, alpha=0.2,color='green')
# axs[1].plot(epochs, gan100_dus_S, label='100GAN',color='red')
# axs[1].fill_between(epochs, gan100_dus_S - gan100_dus_std_S, gan100_dus_S + gan100_dus_std_S, alpha=0.2,color='red')
# axs[1].set_title("Discriminator's uncntrolled Agents' Scores (DUS)\nSDD dataset - Sampler Scheme", wrap=True)
# axs[1].set_xlabel("Discriminator's epoch")
# axs[1].set_ylabel("Score")
# axs[1].legend()
# axs[1].grid(False)

axs[1].plot(epochs, gm10_nomission_S, label='10SM-no mission',  color='green')
axs[1].fill_between(epochs, gm10_nomission_S - gm10_nomission_std_S, gm10_nomission_S + gm10_nomission_std_S, alpha=0.2,  color='green')

axs[1].plot(epochs, gm100_nomission_S, label='100SM-no mission',  color='red')
axs[1].fill_between(epochs, gm100_nomission_S - gm100_nomission_std, gm100_nomission_S + gm100_nomission_std_S, alpha=0.2,  color='red')

axs[1].set_title("(b) Controlled (No Mission) Agents' Scores\nSDD dataset - Sampler Scheme")
axs[1].set_xlabel("Discriminator's epoch")
axs[1].set_ylabel("Score")
axs[1].legend()
axs[1].grid(False)

axs[2].plot(epochs, base_recall, label='Base-Rigid')
axs[2].fill_between(epochs, base_recall - base_recall_std, base_recall + base_recall_std, alpha=0.2)
axs[2].plot(epochs, closest_recall, label='Base-Smooth')
axs[2].fill_between(epochs, closest_recall - closest_recall_std, closest_recall + closest_recall_std, alpha=0.2)
axs[2].plot(epochs, gan10_recall_S, label='10GAN',color='green')
axs[2].fill_between(epochs, gan10_recall_S - gan100_recall_std_S, gan10_recall_S + gan100_recall_std_S, alpha=0.2,color='green')
axs[2].plot(epochs, gan100_recall_S, label='100GAN',color='red')
axs[2].fill_between(epochs, gan100_recall_S - gan100_recall_std_S, gan100_recall_S + gan100_recall_std_S, alpha=0.2,color='red')
axs[2].set_title("(c) Agents' Recall\nSDD dataset - Sampler Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("Recall")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(1, 1, figsize=(8, 5))

axs.plot(epochs, base_dus, label='Base-Rigid')
axs.fill_between(epochs, base_dus - base_dus_std, base_dus + base_dus_std, alpha=0.2)
axs.plot(epochs, closest_dus, label='Base-Smooth')
axs.fill_between(epochs, closest_dus - closest_dus_std, closest_dus + closest_dus_std, alpha=0.2)
axs.plot(epochs, gan10_dus_S, label='10GAN',color='green')
axs.fill_between(epochs, gan10_dus_S - gan10_dus_std_S, gan10_dus_S + gan10_dus_std_S, alpha=0.2,color='green')
axs.plot(epochs, gan100_dus_S, label='100GAN',color='red')
axs.fill_between(epochs, gan100_dus_S - gan100_dus_std_S, gan100_dus_S + gan100_dus_std_S, alpha=0.2,color='red')
axs.set_title("Discriminator's uncontrolled Agents' Scores\nSDD dataset - Sampler Scheme", wrap=True)
axs.set_xlabel("Discriminator's epoch")
axs.set_ylabel("Score")
axs.legend()
axs.grid(False)

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].plot(epochs, base_acc, label='Base-Rigid')
axs[0].fill_between(epochs, base_acc - base_acc_std, base_acc + base_acc_std, alpha=0.2)
axs[0].plot(epochs, closest_acc, label='Base-Smooth')
axs[0].fill_between(epochs, closest_acc - closest_acc_std, closest_acc + closest_acc_std, alpha=0.2)
axs[0].plot(epochs, gan10_acc, label='10GAN' ,color='green')
axs[0].fill_between(epochs, gan10_acc - gan10_acc_std, gan10_acc + gan10_acc_std, alpha=0.2,color='green')
axs[0].plot(epochs, gan100_acc, label='100GAN',color='red')
axs[0].fill_between(epochs, gan100_acc - gan100_acc_std, gan100_acc + gan100_acc_std, alpha=0.2,color='red')
axs[0].set_title("Agents' Accuracy\nSDD dataset - Sampler Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Accuracy")
axs[0].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[0].grid(False)

axs[1].plot(epochs, base_per, label='Base-Rigid')
axs[1].fill_between(epochs, base_per - base_per_std, base_per + base_per_std, alpha=0.2)
axs[1].plot(epochs, closest_per, label='Base-Smooth')
axs[1].fill_between(epochs, closest_per - closest_per_std, closest_per + closest_per_std, alpha=0.2)
axs[1].plot(epochs, gan10_per, label='10GAN',color='green')
axs[1].fill_between(epochs, gan10_per - gan10_per_std, gan10_per + gan10_per_std, alpha=0.2,color='green')
axs[1].plot(epochs, gan100_per, label='100GAN',color='red')
axs[1].fill_between(epochs, gan100_per - gan100_per_std, gan100_per + gan100_per_std, alpha=0.2,color='red')
axs[1].set_title("Agents' Precision\nSDD dataset - Sampler Scheme", wrap=True)
axs[1].set_xlabel("Discriminator's epoch")
axs[1].set_ylabel("Precision")
axs[1].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[1].grid(False)

axs[2].plot(epochs, base_f1, label='Base-Rigid')
axs[2].fill_between(epochs, base_f1 - base_f1_std, base_f1 + base_f1_std, alpha=0.2)
axs[2].plot(epochs, closest_f1, label='Base-Smooth')
axs[2].fill_between(epochs, closest_f1 - closest_f1_std, closest_f1 +closest_f1_std, alpha=0.2)
axs[2].plot(epochs, gan10_f1, label='10GAN',color='green')
axs[2].fill_between(epochs, gan10_f1 - gan10_f1_std, gan10_f1 + gan10_f1_std, alpha=0.2,color='green')
axs[2].plot(epochs, gan100_f1, label='100GAN',color='red')
axs[2].fill_between(epochs, gan100_f1 - gan100_f1_std, gan100_f1 +gan100_f1_std, alpha=0.2,color='red')

axs[2].set_title("Agents' F1 Score\nSDD dataset - Sampler Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("F1")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()
