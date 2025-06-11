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
base_dus = np.array([0.67, 0.86, 0.99, 0.92, 0.98, 0.99])
base_dus_std = np.array([0.05, 0.04, 0.02, 0.03,0.01, 0.01])

base_dcs = np.array([0.4, 0.16, 0.09, 0.14, 0.08, 0.05])
base_dcs_std = np.array([0.03, 0.06, 0.07,0.07, 0.08, 0.08])

base_recall = np.array([0.99, 1, 0.99, 0.99, 0.99, 0.99])
base_recall_std = np.array([0.03,0, 0,0, 0,0, ])
base_acc = np.array([0.72, 0.72, 0.72, 0.72, 0.72, 0.72])
base_acc_std =  np.array([0.14, 0.13, 0.13, 0.13, 0.13, 0.13])

base_per =  np.array([0.65, 0.65, 0.65, 0.65, 0.65, 0.65])
base_per_std =  np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

base_f1 = np.array([0.78, 0.78, 0.78, 0.78, 0.78, 0.78])
base_f1_std =  np.array([0.09, 0.09, 0.09 , 0.09, 0.09, 0.09])




# Data for Closest
closest_dus = np.array([0.69, 0.97, 0.98, 0.99, 0.99, 0.99])
closest_dus_std = np.array([0.03, 0.01, 0.01, 0.0, 0.0, 0.0])

closest_dcs = np.array([0.58, 0.17,0.07, 0.08, 0.09, 0.08])
closest_dcs_std = np.array([0.03, 0.09, 0.09, 0.09, 0.1, 0.09])

closest_recall = np.array([0.43, 0.99,0.99 ,0.99,0.99,0.99])
closest_recall_std = np.array([0.29,0.01, 0.01,0.01, 0.01,0.01, ])

closest_acc = np.array([0.45, 0.72, 0.72, 0.72, 0.72, 0.72])
closest_acc_std =  np.array([0.27, 0.13, 0.13, 0.13, 0.13, 0.13])

closest_per =  np.array([0.42, 0.65, 0.65, 0.65, 0.65, 0.65])
closest_per_std =  np.array([0.29, 0.12, 0.12, 0.12, 0.12, 0.12])

closest_f1 = np.array([0.42, 0.78, 0.78, 0.78, 0.78, 0.78])
closest_f1_std =  np.array([0.29, 0.09, 0.09 , 0.09, 0.09, 0.09])


gan10_dcs = np.array([0.08, 0.07, 0.02, 0.01, 0.02, 0.0])
gan10_dcs_std = np.array([0.03,0.03, 0.02,0.01,0.02,0.0])

gan10_dus = np.array([0.95, 0.97, 0.98, 0.98 ,0.99, 0.99])
gan10_dus_std = np.array([0.02,0.02,0.01, 0.01,0.01,0.01])

gan10_recall = np.array([1, 1, 1, 1, 1, 1])
gan10_recall_std =np.array([0,0.0,0.0, 0.0,0.0,0])
gan10_acc = np.array([0.72, 0.72, 0.72, 0.72, 0.72, 0.72])
gan10_acc_std =  np.array([0.13, 0.13, 0.13, 0.13, 0.13, 0.13])

gan10_per =  np.array([0.65, 0.65, 0.65, 0.65, 0.65, 0.65])
gan10_per_std =  np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

gan10_f1 = np.array([0.78, 0.78, 0.78, 0.78, 0.78, 0.78])
gan10_f1_std =  np.array([0.09, 0.09, 0.09 , 0.09, 0.09, 0.09])



gan100_dcs = np.array([0.49, 0.53,0.32 , 0.36, 0.26, 0.29])
gan100_dcs_std = np.array([0.00,0.02,0.06,0.06,0.06,0.06])

gan100_dus = np.array([0.5, 0.64, 0.73 , 0.79, 0.71, 0.79])
gan100_dus_std = np.array([0.01,0.02, 0.04,0.04,0.05,0.04])

gan100_recall = np.array([0.64, 0.49 , 0.99, 0.98, 1, 0.99])
gan100_recall_std = np.array([0.22,0.27,0.0, 0.04,0,0])
gan100_acc = np.array([0.4, 0.45, 0.72, 0.71, 0.72, 0.72])
gan100_acc_std =  np.array([0.17, 0.27, 0.13, 0.14, 0.13, 0.13])

gan100_per =  np.array([0.41, 0.42, 0.65, 0.65, 0.65, 0.65])
gan100_per_std =  np.array([0.13, 0.29, 0.12, 0.13, 0.12, 0.12])

gan100_f1 = np.array([0.5, 0.42, 0.78, 0.77, 0.78, 0.78])
gan100_f1_std =  np.array([0.16, 0.29, 0.09, 0.09, 0.09 ,0.09])



# Data for 10GM-no mission
gm10_nomission = np.array([0.06, 0.04, 0.01, 0.08, 0.0, 0.0])
gm10_nomission_std = np.array([0.03, 0.02, 0.01, 0.03, 0.0, 0.0])

# Data for 100GM-no mission
gm100_nomission = np.array([0.48, 0.47, 0.22, 0.25, 0.16, 0.18])
gm100_nomission_std = np.array([0.0, 0.02, 0.03, 0.04, 0.03, 0.03])




gan250_dcs =np.array([0.49, 0.48, 0.49, 0.49, 0.49, 0.5])
gan250_dcs_std = np.array([0, 0.01,0.01, 0.01, 0.02, 0.02])

gan250_dus = np.array([0.49, 0.49, 0.51, 0.51, 0.51, 0.54])
gan250_dus_std = np.array([0.0, 0.01, 0.02,0.02, 0.02, 0.02])

gan250_recall = np.array([0.81, 0.97, 0.76, 0.76, 0.73, 0.69])
gan250_recall_std = np.array([0.17, 0.05, 0.17, 0.16, 0.2, 0.17])

gan250_acc = np.array([0.48, 0.5, 0.56, 0.53, 0.52, 0.57])
gan250_acc_std =  np.array([0.14, 0.08, 0.17, 0.15, 0.18, 0.2])

gan250_per = np.array([0.47, 0.53, 0.53, 0.53, 0.5, 0.55])
gan250_per_std =  np.array([0.09, 0.05, 0.14, 0.11, 0.14, 0.17])

gan250_f1 = np.array([0.6, 0.69, 0.62, 0.6, 0.59, 0.61])
gan250_f1_std =  np.array([0.12, 0.04, 0.15 , 0.13, 0.16, 0.17])

gan250_nomission =np.array([0.49, 0.47, 0.47, 0.46, 0.45, 0.45])
gan250_nomission_std = np.array([0.0, 0.01, 0.01, 0.01, 0.01,  0.01])






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
axs[0].set_title("(a) Discriminator's controlled Agents' Scores\nNBA dataset - GroupNet Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Score")
axs[0].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[0].grid(False)



axs[1].plot(epochs, gm10_nomission, label='10GM-no mission',  color='green')
axs[1].fill_between(epochs, gm10_nomission - gm10_nomission_std, gm10_nomission + gm10_nomission_std, alpha=0.2,  color='green')

axs[1].plot(epochs, gm100_nomission, label='100GM-no mission',  color='red')
axs[1].fill_between(epochs, gm100_nomission - gm100_nomission_std, gm100_nomission + gm100_nomission_std, alpha=0.2,  color='red')

axs[1].plot(epochs, gan250_nomission, label='250GM-no mission',  color='purple')
axs[1].fill_between(epochs, gan250_nomission - gan250_nomission_std, gan250_nomission +gan250_nomission_std, alpha=0.2,  color='purple')

axs[1].set_title("(b) Controlled (No Mission) Agents' Scores\nNBA dataset - GroupNet Scheme")
axs[1].set_xlabel("Discriminator's epoch")
axs[1].set_ylabel("Score")
axs[1].legend(loc='lower right', bbox_to_anchor=(0.99, 0.06))
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

axs[2].set_title("(c) Agents' Recall\nNBA dataset - GroupNet Scheme", wrap=True)
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
axs.set_title("Discriminator's uncontrolled Agents' Scores\nNBA dataset - GroupNet Scheme", wrap=True)
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
axs[0].set_title("(A) Agents' Accuracy\nNBA dataset - GroupNet Scheme", wrap=True)
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
axs[1].set_title("(b) Agents' Precision\nNBA dataset - GroupNet Scheme", wrap=True)
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

axs[2].set_title("(c) Agents' F1 Score\nNBA dataset - GroupNet Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("F1")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()


#####SAMPLER#################
# Data for Baseline
base_dus = np.array([0.8, 0.84, 0.93, 0.96, 0.97, 0.92])
base_dus_std = np.array([0.03, 0.03, 0.03, 0.02,0.02, 0.04])

base_dcs = np.array([0.39,0.26,0.15, 0.09, 0.07, 0.05])
base_dcs_std = np.array([0.07, 0.07, 0.08,0.08, 0.09, 0.08])

base_recall = np.array([0.94, 0.99, 0.99, 0.99, 0.99, 1])
base_recall_std = np.array([0.07,0.02, 0.01,0.01, 0.0,0, ])
base_acc = np.array([0.7, 0.72, 0.72, 0.72, 0.72, 0.72])
base_acc_std =  np.array([0.15, 0.13, 0.14, 0.13, 0.13, 0.13])

base_per =  np.array([0.64, 0.65, 0.65, 0.65, 0.66, 0.66])
base_per_std =  np.array([0.14, 0.13, 0.13, 0.13, 0.13, 0.13])

base_f1 = np.array([0.76, 0.78, 0.78, 0.78, 0.78, 0.78])
base_f1_std =  np.array([0.1, 0.09, 0.09, 0.09, 0.09, 0.09])



# Data for Closest
closest_dus = np.array([0.71, 0.94, 0.97, 0.98, 0.99, 0.99])
closest_dus_std = np.array([0.04, 0.02, 0.01, 0.01, 0.0, 0.0])

closest_dcs = np.array([0.48, 0.12, 0.07, 0.08, 0.08, 0.06])
closest_dcs_std = np.array([0.05, 0.09, 0.09, 0.09, 0.09, 0.09])

closest_recall = np.array([0.77, 1,0.99, 0.99, 0.99, 0.99])
closest_recall_std = np.array([0.14,0, 0.0,0.0, 0.0,0.0, ])
closest_acc = np.array([0.62, 0.72, 0.72, 0.72, 0.72, 0.72])
closest_acc_std =  np.array([0.19, 0.13, 0.13, 0.13, 0.13, 0.13])

closest_per =  np.array([0.59, 0.66, 0.66, 0.66, 0.66, 0.6])
closest_per_std =  np.array([0.17, 0.13, 0.13 , 0.13, 0.13, 0.13])

closest_f1 = np.array([0.67, 0.78, 0.78, 0.78, 0.78, 0.78])
closest_f1_std =  np.array([0.15, 0.09, 0.09, 0.09, 0.09, 0.09])


gan10_dcs_S = np.array([0.48, 0.46, 0.43, 0.43, 0.43, 0.42])
gan10_dcs_std_S = np.array([0.02, 0.03, 0.03, 0.03, 0.03, 0.04])

gan10_dus_S = np.array([0.52, 0.53, 0.55, 0.55, 0.56, 0.59])
gan10_dus_std_S = np.array([0.01,0.01, 0.02, 0.02, 0.02, 0.02])

gan10_recall_S = np.array([0.81, 0.93, 0.98, 0.98, 0.98, 0.97])
gan10_recall_std_S =np.array([0.16, 0.08, 0.04, 0.03, 0.03, 0.04])
gan10_acc =np.array([0.61, 0.67, 0.7, 0.7, 0.7, 0.7])
gan10_acc_std =  np.array([0.18, 0.15, 0.14, 0.14, 0.14, 0.14])

gan10_per =  np.array([0.58, 0.62, 0.64, 0.64, 0.64, 0.64])
gan10_per_std =  np.array([0.16 ,0.13, 0.12  , 0.12, 0.12, 0.13])

gan10_f1 = np.array([0.67, 0.74, 0.77, 0.77, 0.77, 0.77])
gan10_f1_std =  np.array([0.15, 0.1, 0.09, 0.09, 0.09, 0.09])



gan100_dcs_S = np.array([0.33, 0.38, 0.2, 0.16, 0.19, 0.14])
gan100_dcs_std_S = np.array([0.03, 0.05, 0.03, 0.03, 0.04, 0.03])

gan100_dus_S = np.array([0.64, 0.66, 0.61, 0.63, 0.66, 0.69])
gan100_dus_std_S = np.array([0.01, 0.02, 0.02, 0.02, 0.01, 0.02])

gan100_recall_S = np.array([1, 0.99, 1, 1, 1, 1])
gan100_recall_std_S = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
gan100_acc = np.array([0.72, 0.72, 0.72, 0.72, 0.72, 0.72])
gan100_acc_std =  np.array([0.13, 0.13, 0.13, 0.13, 0.13, 0.13])

gan100_per =  np.array([0.66, 0.65, 0.66, 0.66, 0.66, 0.6])
gan100_per_std =  np.array([0.13, 0.13, 0.13, 0.13, 0.13, 0.13])

gan100_f1 = np.array([0.78, 0.78, 0.78, 0.78, 0.78, 0.78])
gan100_f1_std =  np.array([0.09, 0.09, 0.09, 0.09, 0.09, 0.09])

# Data for 10GM-no mission
gm10_nomission_S = np.array([0.48, 0.46, 0.43, 0.42, 0.42, 0.42])
gm10_nomission_std_S = np.array([0.02, 0.02, 0.03, 0.03 , 0.03, 0.04])

# Data for 100GM-no mission
gm100_nomission_S = np.array([0.64, 0.66, 0.61, 0.62, 0.65, 0.65])
gm100_nomission_std_S = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02])

fig, axs = plt.subplots(1, 3, figsize=(20, 5))


axs[0].plot(epochs, base_dcs, label='Base-Rigid')
axs[0].fill_between(epochs, base_dcs - base_dcs_std, base_dcs + base_dcs_std, alpha=0.2)
axs[0].plot(epochs, closest_dcs, label='Base-Smooth')
axs[0].fill_between(epochs, closest_dcs - closest_dcs_std, closest_dcs + closest_dcs_std, alpha=0.2)
axs[0].plot(epochs, gan10_dcs_S, label='10GAN',color='green')
axs[0].fill_between(epochs, gan10_dcs_S - gan10_dcs_std_S, gan10_dcs_S + gan10_dcs_std_S, alpha=0.2,color='green')
axs[0].plot(epochs, gan100_dcs_S, label='100GAN',color='red')
axs[0].fill_between(epochs, gan100_dcs_S - gan100_dcs_std_S, gan100_dcs_S + gan100_dcs_std_S, alpha=0.2,color='red')
axs[0].set_title("(a) Discriminator's controlled Agents' Scores\nNBA dataset - Sampler Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Score")
axs[0].legend()
axs[0].grid(False)



axs[1].plot(epochs, gm10_nomission_S, label='10SM-no mission',  color='green')
axs[1].fill_between(epochs, gm10_nomission_S - gm10_nomission_std_S, gm10_nomission_S + gm10_nomission_std_S, alpha=0.2,  color='green')

axs[1].plot(epochs, gm100_nomission_S, label='100SM-no mission',  color='red')
axs[1].fill_between(epochs, gm100_nomission_S - gm100_nomission_std, gm100_nomission_S + gm100_nomission_std_S, alpha=0.2,  color='red')

axs[1].set_title("(b) Controlled (No Mission) Agents' Scores\nNBA dataset - Sampler Scheme")
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
axs[2].set_title("(c) Agents' Recall\nNBA dataset - Sampler Scheme", wrap=True)
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
axs.set_title("Discriminator's uncontrolled Agents' Scores\nNBA dataset - Sampler Scheme", wrap=True)
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
axs[0].set_title("(a) Agents' Accuracy\nNBA dataset - Sampler Scheme", wrap=True)
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
axs[1].set_title("(b) Agents' Precision\nNBA dataset - Sampler Scheme", wrap=True)
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

axs[2].set_title("(c) Agents' F1 Score\nNBA dataset - Sampler Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("F1")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()