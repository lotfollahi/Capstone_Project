import classify
import matplotlib.pyplot as plt
from envi import envi
import scipy.misc

prob_file = 'cnn-load-batch-test-response'
output_file = 'cnn-load-batch-test.png'

prob_envi= envi(prob_file)
prob_image = prob_envi.loadall()

class_image = classify.prob2class(prob_image)
rgb = classify.class2color(class_image)
scipy.misc.imsave(output_file, rgb)
