import keras
import tensorflow as tf

class RepModel(keras.Model):
	def __init__(self):
		super().__init__()

		#first convolutional layer
		self.conv1 = keras.layers.Conv2D(256, kernel_size=(2,2), strides=(2,2), input_shape=(64,64,3), activation='relu', data_format="channels_last")

		#second convolutional layer, combine with a residual from conv1
		self.conv2 = keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', data_format="channels_last")

		#third convolutional layer
		self.conv3 = keras.layers.Conv2D(256, kernel_size=(2,2), strides=(2,2),activation='relu', data_format="channels_last")
    
        #fourth convolutional layer, combine with: concat{ residual from conv3, broadcast(v) }
        self.conv4 = keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1),activation='relu', data_format="channels_last")
        
        #fifth convolutional layer
        self.conv5 = keras.layers.Conv2D(256, kernel_size=(3,3), strides=(1,1),activation='relu', data_format="channels_last")
        
        #sixth convolutional layer
        self.conv6 = keras.layers.Conv2D(256, kernel_size=(1,1), strides=(1,1),activation='relu', data_format="channels_last")

	def call(self, x, v):
		a1 = self.conv1(x)

		z2 = self.conv2(a1)
		a2 = keras.layers.add([a1,z2]) #adding back the residual layer

		a3 = self.conv3(a2)

        #concatenate the broadcasted v
		c = keras.layers.concatenate([tf.broadcast_to(v, [16,16,7]), a3], axis=-1)

        z4 = self.conv4(c)
        a4 = keras.layers.add([c,z4])

        a5 = self.conv5(a4)

        a6 = self.conv6(a5)


## test
rep = RepModel()
image = tf.image.encode
