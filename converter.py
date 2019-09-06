import numpy as np
from os import listdir
import glm
import math
import matplotlib.pyplot as plt

frame_time = 1.0/30.0
end_point_indices = [0, 3, 15, 19, 21, 22, 23, 24]
hand_indices = [21, 22, 23, 24]
heights = [1.7, 1.7, 1.4, 1.2, 1.2, 0.8, 0.5, 1.4, 0.8, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.7, 2.5]

trainP = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]

from IPython import embed

plot_lists = np.zeros((3, 110, 75))

def inverse_quat(q1):
	return glm.vec4([q1[0], -q1[1], -q1[2], -q1[3]])

def multiply_quat(q1, q2):
	w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
	x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
	y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
	z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
	return glm.vec4([w, x, y, z])

def convert_data(address, write, S, name=""):
	print(address)
	file = open(address)
	lines = file.readlines()	

	frame_count = int(lines[0])
	class_num = int(address[-11:-9])
	bodyinfo=[] # to store multiple skeletons per frame
	index = 1
	out = np.zeros([frame_count, 77])
	#out1 = np.zeros([frame_count, 77 + 4 * 3 + 2 * 2])
	out1 = np.zeros([frame_count, 77])
	data = np.zeros([frame_count, 75 + 3])
	data1 = np.zeros([frame_count, 75 + 3 ])
	last_root_pos = glm.vec3([0.0, 0.0, 0.0])
	root_last_Z = glm.vec3([0.0, 0.0, 1.0])
	root_quat = glm.vec4([0.0, 0.0, 0.0, 0.0])
	root_pos = glm.vec3([0.0, 0.0, 0.0])
	joint_average_velocities = []
	initial_id = 0
	for i in range (0, 24): 
		joint_average_velocities = joint_average_velocities + [0.0]
	for i in range (0, frame_count):
		body_count = int(lines[index]) # no of observerd skeletons in current frame
		index = index + 1
		if i == 0 and body_count == 0:
			return out, out1, False
		for j in range (0, body_count):
			id_line = lines[index].split(" ")
			character_id = int(id_line[0])
			if i == 0 and j == 0:
				initial_id = character_id
			index = index + 1
			
			number_of_joints =  int(lines[index])
			M = glm.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
			index = index + 1
			

			for k in range(0, number_of_joints):
				joint_line = lines[index].split(" ")
				index = index + 1
				if k == 0 : 				
					if character_id == initial_id:
						root_pos = glm.vec3([float(joint_line[0]), float(joint_line[1]), float(joint_line[2])])
						root_quat = glm.vec4([float(joint_line[-5]), float(joint_line[-4]), float(joint_line[-3]), float(joint_line[-2])])
						X = glm.vec4(0.0, 1.0, 0.0, 0.0)
						Y = glm.vec4(0.0, 0.0, 1.0, 0.0)
						Z = glm.vec4(0.0, 0.0, 0.0, 1.0) 
						newX = multiply_quat(root_quat, multiply_quat(X, inverse_quat(root_quat)))
						newY = multiply_quat(root_quat, multiply_quat(Y, inverse_quat(root_quat)))
						newZ = multiply_quat(root_quat, multiply_quat(Z, inverse_quat(root_quat)))
						#M = glm.mat3[[newX(0), newY(0), newZ(0)], [newX(1), newY(1), newZ(1)], [newX(2), newY(2), newZ(2)]]
						#print(newX[0], newY, newZ)					
						M = glm.mat3([[newX[1], newY[1], newZ[1]], [newX[2], newY[2], newZ[2]], [newX[3], newY[3], newZ[3]]])
						root_linear_velocity = glm.vec3(root_pos[0] - last_root_pos[0], root_pos[1] - last_root_pos[1], root_pos[2] - last_root_pos[2]) / frame_time
						velocity_in_frame = M * root_linear_velocity
						#print(velocity_in_frame)
						
						out[i,77-5] = velocity_in_frame[0]
						out[i,77-4] = velocity_in_frame[1]
						out[i,77-3] = velocity_in_frame[2]						

						out1[i,77-5] = velocity_in_frame[0]
						out1[i,77-4] = velocity_in_frame[1]
						out1[i,77-3] = velocity_in_frame[2]						

						data[i, 0] = float(joint_line[0])
						data[i, 1] = float(joint_line[1])
						data[i, 2] = float(joint_line[2])
						
						data1[i, 0] = float(joint_line[0])
						data1[i, 1] = float(joint_line[1])
						data1[i, 2] = float(joint_line[2])
						
						newZ = glm.vec3(newZ[1], newZ[2], newZ[3])

						tempZ = glm.vec3(newZ[0], 0.0, newZ[2])
						tempZ = tempZ / glm.length(tempZ)					
						
						temp_lastZ = glm.vec3(root_last_Z[0], 0.0, root_last_Z[2])
						temp_lastZ = temp_lastZ / glm.length(temp_lastZ)
	
						if glm.length(glm.cross(temp_lastZ, tempZ)) > 0.001 :
							out[i,77-1] = (glm.cross(temp_lastZ, tempZ)/glm.length(glm.cross(temp_lastZ, tempZ)))[1] * math.acos(glm.dot(temp_lastZ, tempZ))/frame_time
							out1[i,77-1] = (glm.cross(temp_lastZ, tempZ)/glm.length(glm.cross(temp_lastZ, tempZ)))[1] * math.acos(glm.dot(temp_lastZ, tempZ))/frame_time
						else:
							out[i,77-1] = 0
							out1[i,77-1] = 0
						out[i,77-2] = root_pos[1] - heights[S-1]
						out1[i,77-2] = root_pos[1] - heights[S-1]
						
						last_root_pos = glm.vec3(root_pos[0], root_pos[1], root_pos[2])
						root_last_Z = newZ
						data[i, -3] = newZ[0]
						data[i, -2] = newZ[1]
						data[i, -1] = newZ[2]
						
						data1[i, -3] = newZ[0]
						data1[i, -2] = newZ[1]
						data1[i, -1] = newZ[2]
						

				else : 
					if character_id == initial_id:
						joint_pos = glm.vec3([float(joint_line[0]), float(joint_line[1]), float(joint_line[2])])
						joint_local_pos = glm.vec3([joint_pos[0] - root_pos[0], joint_pos[1] - root_pos[1], joint_pos[2] - root_pos[2]])
						joint_local_pos = M * joint_local_pos
											
						out[i, 3 * (k-1)] = joint_local_pos[0]
						out[i, 3 * (k-1) + 1] = joint_local_pos[1]
						out[i, 3 * (k-1) + 2] = joint_local_pos[2]

						out1[i, 3 * (k-1)] = joint_local_pos[0]
						out1[i, 3 * (k-1) + 1] = joint_local_pos[1]
						out1[i, 3 * (k-1) + 2] = joint_local_pos[2]

						data[i, 3 * k] = float(joint_line[0])
						data[i, 3 * k + 1] = float(joint_line[1])
						data[i, 3 * k + 2] = float(joint_line[2])
						
						data1[i, 3 * k] = float(joint_line[0])
						data1[i, 3 * k + 1] = float(joint_line[ 1])
						data1[i, 3 * k + 2] = float(joint_line[2])
						
						if i!= 0:
							velocity_vector = glm.vec3([data1[i, 3 * k] - data1[i-1, 3 * k], data1[i, 3 * k + 1] - data1[i-1, 3 * k + 1], data1[i, 3 * k + 2] - data1[i-1, 3 * k + 2]]) / frame_time
							current_velocity = glm.length (velocity_vector)
							#print(current_velocity , joint_average_velocities[k-1] )
							if current_velocity > 1.5 * joint_average_velocities[k-1] and i > 1: 
								newPos = glm.vec3([data1[i-1, 3 * k], data1[i-1, 3 * k + 1], data1[i-1, 3 * k + 2]])	
								newPos = newPos + velocity_vector * frame_time / glm.length(velocity_vector) * 1.5 * joint_average_velocities[k-1]
								#data1[i, 3 * k] = (data1[i, 3 * k] / 30.0 + c * data1[i-1, 3 * k]) * (1/(c + 1.0/30.0))
								#data1[i, 3 * k + 1] = (data1[i, 3 * k + 1] / 30.0 + c * data1[i-1, 3 * k + 1]) * (1/(c + 1.0/30.0)) 
								#data1[i, 3 * k + 2] = (data1[i, 3 * k + 2] / 30.0 + c * data1[i-1, 3 * k + 2]) * (1/(c + 1.0/30.0))
								data1[i, 3 * k] = newPos[0]
								data1[i, 3 * k + 1] = newPos[1]
								data1[i, 3 * k + 2] = newPos[2]
	
								joint_average_velocities[k-1] = joint_average_velocities[k-1] * 1.5								
							else :
								c1 = 2.0
								c2 = 1.0
								joint_average_velocities[k-1] = (joint_average_velocities[k-1] * c1 + current_velocity * c2) / (c1 + c2) 				
							modified_pos = M * glm.vec3([data1[i, 3 * k] - root_pos[0], data1[i, 3 * k + 1] - root_pos[1], data1[i, 3 * k + 2] - root_pos[2]])

							out1[i, 3 * (k-1)] = modified_pos[0]
							out1[i, 3 * (k-1) + 1] = modified_pos[1]
							out1[i, 3 * (k-1) + 2] = modified_pos[2]
					
			
	out[0,77-5:77] = out[1,77-5:77]
	out1[0,77-5:77] = out1[1,77-5:77]
	m = np.mean(out)
	s = np.std(out)
	if (name == "S001C001P001R001A059" or name == "S001C002P001R001A059" or name == "S001C003P001R001A059"):
		view = int(name[5:8])
		plot_lists[view-1,0:out1.shape[0],:] = out1[:,0:75]
			
		
	#if write == True and (name == "S001C001P001R001A001" or name == "S001C001P001R001A002" or name == "S001C001P001R001A003" or name == "S001C001P001R001A004" or name == "S001C001P001R001A006" or name == "S001C001P001R001A007") :
	if write == True and (name == "S001C001P001R001A001" or name == "S001C001P001R001A002" or name == "S001C001P001R001A003" or name == "S001C001P001R001A059" or name == "S001C002P001R001A001" or name == "S001C002P001R001A002" or name == "S001C002P001R001A003" or name == "S001C002P001R001A059" or name == "S001C003P001R001A001" or name == "S001C003P001R001A002" or name == "S001C003P001R001A003" or name == "S001C003P001R001A059"):
		file = open("animations/" + name + "/data.js", "w")
		s = "var data = [\n"		
		for i in range(0, out.shape[0]):
			
		
			for j in range(0, 77):
				s = s + str(out[i, j]) + ","
		s = s + "\n" + "]"	
		file.write(s)

		file = open("animations/" + name + "3/data.js", "w")
		s = "var data = [\n"		
		for i in range(0, out1.shape[0]):
			
		
			for j in range(0, 77):
				s = s + str(out1[i, j]) + ","
		s = s + "\n" + "]"	
		file.write(s)
		
		file = open("animations/" + name + "1/data.js", "w")
		s = "var data = [\n"		
		for i in range(0, data.shape[0]):
			for j in range(0, 78):
				s = s + str(data[i, j]) + ","
		s = s + "\n" + "]"	
		file.write(s)

		file = open("animations/" + name + "2/data.js", "w")
		s = "var data = [\n"		
		for i in range(0, data.shape[0]):
			for j in range(0, 78):
				s = s + str(data1[i, j]) + ","
		s = s + "\n" + "]"	
		file.write(s)

	if m != 0.0 or s!= 0.0:
		return out, out1, True
	else :	
		return out, out1, False

def convert_folder(folder, folder1, folder2):
	files = sorted(listdir(folder))
	write = True
	counter = 0
	sums = np.zeros([1,77])
	sums1 = np.zeros([1,77])
	frame_counts = 0
	for file in files :
		
		S = int(file[1:4])
		P = int(file[9:12])
		name = file[0:20]
		is_test = True		
		for m in range (0,20):
			if trainP[m] == P:
				is_test = False	
		if is_test is False:
			out, out1, state = convert_data(folder + "/" + file, write, S, name)
			if state == True :
				sums = sums + np.sum(out, axis = 0)
				sums1 = sums1 + np.sum(out1, axis = 0)			
				frame_counts = frame_counts + out.shape[0]					

	means = sums / frame_counts
	means1 = sums1 / frame_counts
	embed()
	#means = np.mean(outs, 0)
	#stds = np.std(outs, 0)
	counter = 0
	sums2 = np.zeros([1,77])
	sums21 = np.zeros([1,77])
	for i in range(0,75):
		plt.plot(plot_lists[0,:,i])
		plt.plot(plot_lists[1,:,i])
		plt.plot(plot_lists[2,:,i])
		#plt.show()
		plt.savefig("plots/" + str(i) + ".png" )
		#plt.show()
		plt.close()
		
	for file in files :
		
		S = int(file[1:4])
		P = int(file[9:12])
		
		is_test = True		
		for m in range (0,20):
			if trainP[m] == P:
				is_test = False	
		if is_test is False :
			out, out1, state = convert_data(folder + "/" + file, write, S)
				
			if state == True :
				for i in range (0, out.shape[0]):
					out[i] = out[i] - means
					out1[i] = out1[i] - means1
				out = np.multiply(out, out)
				out1 = np.multiply(out1, out1)					
				sums2 = sums2 + np.sum(out, axis = 0)
				sums21 = sums21 + np.sum(out1, axis = 0)


	stds = np.sqrt(sums2 / frame_counts)
	stds1 = np.sqrt(sums21 / frame_counts)

	embed()

	counter = 0

	for file in files :
		#print (folder + "/" + file)
		#data = read_bvh(folder + "/" + file)
		S = int(file[1:4])
		out, out1, state = convert_data(folder + "/" + file, write, S)

		counter = counter + 1
		if state == True : 
			for i in range (0, out.shape[0]):
				out[i] = np.divide(out[i] - means, stds)
				out1[i] = np.divide(out1[i] - means1, stds1)
				
			np.save(folder1 + "/" + file[0:-4], out)
			np.save(folder2 + "/" + file[0:-4], out1)
		if (counter > 60):
			write = False
	np.save(folder1 + "/means", means)		
	np.save(folder2 + "/means", means1)
	np.save(folder1 + "/stds", stds)		
	np.save(folder2 + "/stds", stds1)		

#a = glm.vec3([1,2,3])
#print(a[0])W	
#convert_data("../nturgb+d_skeletons/S001C001P001R001A008.skeleton")						
convert_folder("../nturgb+d_skeletons", "../ntupositions", "../ntupositions_noise_deducted")

#def convert_all_data(folder1, folder2):
	
