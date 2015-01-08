import numpy.random as npr
import numpy as np

from gol.goal.utilities import collect_training_data, load_images, create_aop_learner
from gol.blockmatching.blockmatching import all_files
from gol.goal.utilities import filter_image
import matplotlib.pyplot as plt
import time

import shelve

if __name__ == '__main__':

    workbench = shelve.open('denoising_workbench.db')

    img_path = "data"
    block = (8, 8)
    step = (4, 4)

    database = [x.bgr2rgb() for x in load_images(all_files(img_path), block, step) ]
    training_data = collect_training_data(database,
                                          lambda x: x.image_block).astype(np.float32)

    aop_training_data_r = collect_training_data(database,lambda x: x.r_block)
    aop_training_data_g = collect_training_data(database,lambda x: x.g_block)
    aop_training_data_b = collect_training_data(database,lambda x: x.b_block)

    nu = 1E3
    kappa = 1 * 1e6
    mu = 8 * 1E2
    lifting = 2
    aop_learner_r, Omega_r, Omega_r_sym = create_aop_learner(aop_training_data_r,lifting, nu, kappa, mu)
    aop_learner_g, Omega_g, Omega_g_sym = create_aop_learner(aop_training_data_g,lifting, nu, kappa, mu)
    aop_learner_b, Omega_b, Omega_b_sym = create_aop_learner(aop_training_data_b,lifting, nu, kappa, mu)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.ion()
    plt.show()

    def constraint_monitor_r(D):
        print "Cost: ", D['c']
        image = filter_image(Omega_r.get_value().T,1)
        image=np.concatenate([image,image,image],axis=2)
        plt.imshow(image)
        plt.pause(0.01)

    start = time.clock()
    aop_learner_r.optimize(1500,constraint_monitor_r,50)
    end = time.clock()
    print "Total elapsed: ", end-start

    def constraint_monitor_g(D):
        print "Cost: ", D['c']
        image = filter_image(Omega_g.get_value().T,1)
        image=np.concatenate([image,image,image],axis=2)
        ax.imshow(image)
        plt.pause(0.01)

    aop_learner_g.optimize(1500,constraint_monitor_g,50)

    def constraint_monitor_b(D):
        print "Cost: ", D['c']
        image = filter_image(Omega_b.get_value().T,1)
        image = np.concatenate([image,image,image],axis=2)
        ax.imshow(image)
        plt.pause(0.01)

    aop_learner_b.optimize(1500,constraint_monitor_b,50)

    workbench['Omega']=[Omega_r.get_value(),Omega_g.get_value(),Omega_b.get_value()]
    workbench['Omega_param']={'lifting':lifting,'nu':nu,'kappa':kappa,'mu':mu}

    workbench.close()