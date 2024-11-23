import numpy as np
import matplotlib.pyplot as plt
import os


path_to_mesh = 'C:/Users/Utilisateur/OneDrive/Bureau/new_meshes'
mesh_ext = '.mesh_005_02_2_ext3_sym.FEM'
mesh_type = 'vv'
S = 3

expo = -8


R = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}rr_S{s:04d}"+mesh_ext) for s in range(S)]).reshape(-1)
Z = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}zz_S{s:04d}"+mesh_ext) for s in range(S)]).reshape(-1)
W = np.hstack([np.fromfile(path_to_mesh+f"/{mesh_type}weight_S{s:04d}"+mesh_ext) for s in range(S)]).reshape(-1)

def change(initial_array):
    new_array = []
    for i,elm in enumerate(initial_array):
        new_array.append([elm,i])
    return np.array(new_array)

def compare(elm1,elm2):
    if elm1[0] < elm2[0]:
        return np.array([elm1,elm2])
    else:
        return np.array([elm2,elm1])

def sort_from_sorted(Z1,Z2):
    i,j = 0,0
    a = True
    final_z = []
    #print(np.shape(Z1))
    while a:
       # print(Z2)
        if Z1[i][0] < Z2[j][0]:
            final_z.append(Z1[i])
            i += 1
        else:
            final_z.append(Z2[j])
            j += 1
        if i == len(Z1):
            for elm in Z2[j:]:
                final_z.append(elm)
            a = False
        if j == len(Z2):
            for elm in Z1[i:]:
                final_z.append(elm)
            a = False
    return np.array(final_z)


def sort_function(Z):
    if len(Z) == 1:
        return Z
    elif len(Z) == 2:
        return compare(Z[0],Z[1])
    else:
        return sort_from_sorted(sort_function(Z[:len(Z)//2]),sort_function(Z[len(Z)//2:]))

list_z = change(Z)

sorted_z = sort_function(list_z)
for i in range(len(sorted_z)-1):
    assert sorted_z[i,0]<=sorted_z[i+1,0]
# print(np.shape(sorted_z))
# print(np.min(sorted_z[:][0]))
#print(sorted_z[:,0])
# plt.plot(np.arange(len(sorted_z[:,0])),sorted_z[:,0])
# plt.show()
# plt.plot(np.arange(len(sorted_z[:,1])),sorted_z[:,1])
# plt.show()
# for elm in sorted_z:
#      print(elm[0])

list_sorted_z = []
for i in range(len(sorted_z)-1):
    list_sorted_z.append(sorted_z[i+1,0]-sorted_z[i,0])

# plt.plot(np.arange(len(list_sorted_z)),list_sorted_z)
# plt.xlim(0,100)
# #plt.ylim(0,10**(-7))
# plt.show()
def find_symmetric(r,list_indexes_negative_z,list_indexes_positive_z):
 #   print(np.shape(r))
   # print(list_indexes_negative_z)
    sorted_negative_r = sort_function(np.array(r)[np.array(list_indexes_negative_z),:])
    sorted_positive_r = sort_function(np.array(r)[np.array(list_indexes_positive_z),:])
  #  print(sorted_negative_r,sorted_positive_r)
    list_pairs = []
    for i in range(len(sorted_negative_r)):
        list_pairs.append([int(sorted_negative_r[i,1]),int(sorted_positive_r[i,1])])
  #  print(list_pairs)
    return list_pairs

r = change(R)

index = 0
negative_index = len(R)-1
z = sorted_z[0,0]

list_pairs = []

while index < negative_index and z < 0:
    first_index = index
    while np.abs(sorted_z[index+1,0]-sorted_z[index,0]) < 10**expo:
        index += 1
    last_negative_index = len(R)-index-1
  #  print(sorted_z[index,0],-sorted_z[last_negative_index,0])
    #assert sorted_z[index,0]//10**expo == -sorted_z[last_negative_index,0]//10**expo
    assert np.abs(sorted_z[index,0] + sorted_z[last_negative_index,0]) < 10**expo
    list_negative_indexes = []
    list_positive_indexes = []
    for i in range(first_index,index+1):
        list_negative_indexes.append(int(sorted_z[i,1]))
        list_positive_indexes.append(int(sorted_z[-i-1,1]))
    # for i in range(len(list_negative_indexes)):
    #     list_negative_indexes[i] = int(list_negative_indexes[i])
    #     list_positive_indexes[i] = int(list_positive_indexes[i])

    negative_index = last_negative_index-1
    index += 1

    z = sorted_z[first_index,0]
    new_pairs = find_symmetric(r,list_negative_indexes,list_positive_indexes)
    # for elm in new_pairs:
    #     i,j = elm[0],elm[1]
    #     print(R[i],R[j],Z[i],Z[j])
    #     print(new_pairs)
    for elm in new_pairs:
        i,j = elm[0],elm[1]       
        assert np.abs(R[i]-R[j])<10**expo
    list_pairs += new_pairs
    if index%100 == 0:
        print(index)
   # print(index,len(list_pairs),index+last_negative_index,len(new_pairs)-len(list_negative_indexes))
np.save(path_to_mesh+'/list_pairs_vv',np.array(list_pairs))
 #   assert (1+index)*2 == len(list_pairs)
list_differences_R = []
list_differences_Z = []
for elm in list_pairs:
    i,j = elm[0],elm[1]
    list_differences_R.append(np.abs(R[i]-R[j]))
    list_differences_Z.append(np.abs(Z[i] + Z[j]))
plt.plot(np.arange(len(list_differences_R)),list_differences_R,label='R')
plt.plot(np.arange(len(list_differences_Z)),list_differences_Z,label='Z')
plt.legend()
plt.show()