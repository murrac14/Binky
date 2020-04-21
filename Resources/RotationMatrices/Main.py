import numpy
import math


def find_frame_transformation(theta, d, a, alpha):
    # Fill the matrices with relevant angles and translations
    rotation_z = numpy.array([[math.cos(theta), -math.sin(theta), 0, 0],
                              [math.sin(theta), math.cos(theta), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    print("Rotation theta: ")
    print(rotation_z)

    translation_z = numpy.array([[1.0, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, d],
                                 [0, 0, 0, 1]])
    print("Translation d: ")
    print(translation_z)

    translation_x = numpy.array([[1.0, 0, 0, a],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
    print("Translation a: ")
    print(translation_x)

    rotation_x = numpy.array([[1.0, 0, 0, 0],
                              [0, math.cos(alpha), -math.sin(alpha), 0],
                              [0, math.sin(alpha), math.cos(alpha), 0],
                              [0, 0, 0, 1]])
    print("Rotation alpha: ")
    print(rotation_x)

    combined_matrix = numpy.matmul(numpy.matmul(rotation_z, translation_z), numpy.matmul(translation_x, rotation_x))
    print("Combined transformation: ")
    print(combined_matrix)

    # Return the new coordinate frame
    return combined_matrix


def find_6_joint_fkin(dh_table):
    frame_result = numpy.eye(4)
    for joint in range(6):
        print(joint)
        frame_result = numpy.matmul(frame_result, find_frame_transformation(dh_table[joint * 4 + 0],
                                                                            dh_table[joint * 4 + 1],
                                                                            dh_table[joint * 4 + 2],
                                                                            dh_table[joint * 4 + 3]))
    return frame_result


'''
joint_angle_1 = math.radians(float(input("Type in the rotation about the first joint: ")))
joint_angle_2 = math.radians(float(input("Type in the rotation about the second joint: ")))
joint_angle_3 = math.radians(float(input("Type in the rotation about the third joint: ")))
joint_angle_4 = math.radians(float(input("Type in the rotation about the fourth joint: ")))
joint_angle_5 = math.radians(float(input("Type in the rotation about the fifth joint: ")))
joint_angle_6 = math.radians(float(input("Type in the rotation about the sixth joint: ")))
'''

joint_angle_1 = math.radians(-60.8)
joint_angle_2 = math.radians(36)
joint_angle_3 = math.radians(-155)
joint_angle_4 = math.radians(-22.4)
joint_angle_5 = math.radians(24)
joint_angle_6 = math.radians(-42.6)

dh_parameters = [joint_angle_1, 0, 0, math.pi / 2,
                 joint_angle_2, 0, 0.4318, 0,
                 joint_angle_3, 0.15, 0.0203, -math.pi / 2,
                 joint_angle_4, 0.4318, 0, math.pi / 2,
                 joint_angle_5, 0, 0, -math.pi / 2,
                 joint_angle_6, 0, 0, 0]

end_effector_frame = find_6_joint_fkin(dh_parameters)
end_effector_vector = numpy.matmul(end_effector_frame, numpy.array([[0], [0], [0], [1]]))

print("End effector frame relative to the base frame is: ")
print(end_effector_frame)
print("End effector vector relative to the base frame is: ")
print(end_effector_vector)
'''
frame_result = numpy.eye(4)
for joint in range(6):
    print(joint)
    frame_result = numpy.matmul(frame_result, find_frame_transformation(dh_parameters[joint * 4 + 0],
                                                                        dh_parameters[joint * 4 + 1],
                                                                        dh_parameters[joint * 4 + 2],
                                                                        dh_parameters[joint * 4 + 3]))



frame_six_to_zero = frame_result
print("Six to Zero: ")
print(frame_six_to_zero)
frame_zero_to_six = numpy.linalg.inv(frame_result)
print("Zero_to_six: ")
print(frame_zero_to_six)

original_vector = numpy.array([[0], [0], [0], [1]])

end_effector_frame = numpy.matmul(frame_zero_to_six, original_vector)
print("The end-effector frame is located at: ")
print(end_effector_frame)

rotate_z = math.radians(float(input("Enter z rotation: ")))
translate_z = float(input("Enter z translation: "))
translate_x = (float(input("Enter x translation: ")))
rotate_x = math.radians(float(input("Enter x rotation: ")))


# Pre-multiply frame 1 by this to get frame 0
frame_one_to_zero = find_frame_transformation(rotate_z, translate_z, translate_x, rotate_x)
end_effector_frame = numpy.matmul(frame_one_to_zero, origin_matrix)
print("The end effector's frame is: ")
print(end_effector_frame)

# Pre-multiply frame 0 by this to get frame 1
frame_zero_to_one = numpy.linalg.inv(frame_one_to_zero)
print("Inverted: ")
print(frame_zero_to_one)

end_effector_frame = numpy.matmul(frame_zero_to_one, origin_matrix)
print("The end effector's 1st frame is: ")
print(end_effector_frame)

frame_two_to_one = find_frame_transformation(rotate_z, translate_z, translate_x, rotate_x)
frame_one_to_two = numpy.linalg.inv(frame_two_to_one)
frame_zero_to_two = numpy.matmul(frame_zero_to_one, frame_one_to_two)

end_effector_frame = numpy.matmul(frame_zero_to_two, origin_matrix)
print("The end effector's 2nd frame is: ")
print(end_effector_frame)'''
