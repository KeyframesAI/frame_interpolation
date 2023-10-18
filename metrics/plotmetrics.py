import matplotlib.pyplot as plt 
import json

with open("metrics.json") as metricsfile:
    metrics = json.load(metricsfile)
        

x = [float(m['epoch']) + float(m['batch'])/1000.0 for m in metrics] 
g = [float(m['g_loss']) for m in metrics] 
dr = [float(m['real_loss']) for m in metrics] 
df = [float(m['fake_loss'])for m in metrics] 
d = [(float(m['real_loss']) + float(m['fake_loss']))/2.0 for m in metrics] 
plt.plot(x, g, label = "g_loss") 
plt.plot(x, d, label = "d_loss") 
plt.plot(x, dr, label = "d_real_loss") 
plt.plot(x, df, label = "d_fake_loss") 

plt.legend() 
plt.xlabel('epoch') 
plt.title('frame_interpolation_GAN (DCGAN)') 

plt.show() 
