import xml.etree.ElementTree as ET
import xml.dom.minidom
import random
import numpy as np

#### object の名前リスト
object_name = ["back_1", "back_2", "backpack_1", "backpack_2", "boots", "box", "cylinder", "glasse_1", "glasse_2", "grobe", "shoes", "shelv", "suitscase", "teddy", "toy"]

### [幅 or 高さ, 最大倍率]
object_data = {"back_1":[0.0124, 0.0721, 1.6], "back_2":[0.0124, 0.0481, 1.95], "backpack_1":[0.3, 0.2, 1.3], "backpack_2":[0.3, 0.24, 1.3], "boots":[0.256, 0.184, 1.6], "box":[0.0628, 0.111, 1.6], "cloth":[0.0682, 0.711, 1.], \
                "coat":[0.0967, 1.04, 1.], "coats":[0.336, 0.745, 1], "cylinder":[0.0883, 0.181, 1.6], "glasse_1":[0.0352, 0.124, 1.6], "glasse_2":[0.0644, 0.164, 1.6], \
                "grobe":[0.017, 0.0878, 2.], "shoes":[0.0644, 0.121, 1.6], "shelv":[0.337, 0.178, 1.], "suitcase":[0.445, 0.366, 1.], "teddy":[0.242, 0.253, 1.6], "toy":[0.0468, 0.222, 1.6]}

closet_y = [0.7695, 0.855, 1.026, 1.197, 1.368]
scale = [9, 10, 12, 14, 16]

def edit_urdf_real(scale_id, asset_number, handle_z_number, door_wide_base = 0.282):
    object_name = ["back_1", "back_2", "backpack_1", "backpack_2", "boots", "box", "cylinder", "glasse_1", "glasse_2", "grobe", "shoes", "shelve", "suitscase", "teddy_1", "toy"]

### [幅 or 高さ, 最大倍率]
    object_data = {"back_1":[0.0124, 0.0721, 1.6], "back_2":[0.0124, 0.0481, 1.95], "backpack_1":[0.3, 0.2, 1.3], "backpack_2":[0.3, 0.24, 1.3], "boots":[0.256, 0.184, 1.6], "box":[0.0628, 0.111, 1.6], "cloth":[0.0682, 0.711, 1.], \
                "coat":[0.0967, 1.04, 1.], "coats":[0.336, 0.745, 1], "cylinder":[0.0883, 0.181, 1.6], "glasse_1":[0.0352, 0.124, 1.6], "glasse_2":[0.0644, 0.164, 1.6], \
                "grobe":[0.017, 0.0878, 2.], "shoes":[0.0644, 0.121, 1.6], "shelve":[0.337, 0.178, 1.], "suitscase":[0.445, 0.366, 1.], "teddy_1":[0.242, 0.253, 1.6], "toy":[0.0468, 0.222, 1.6]}

    closet_y = [0.7695, 0.855, 1.026, 1.197, 1.368]
    scale = [9, 10, 12, 14, 16]

    ####### handle の高さの設定
    scale = scale_id
    handle_z = handle_z_number

    tree = ET.parse(f'/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets/urdf/closet_for_real/closet_various_handleposition/closet_10.urdf')
    root = tree.getroot()

    handle_y_1 = -1 * ((door_wide_base * scale  / 2) + 0.0075 * scale)
    handle_y_2 = -1 * ((door_wide_base * scale  / 2) - 0.02 * scale)
    handle_y_3 = -1 * ((door_wide_base * scale  / 2))
    
    new_xyz_1 = f'0.075 {handle_y_1+ 0.07}  {handle_z - 0.0411}'
    new_xyz_2 = f'0.075 {handle_y_2} {handle_z - 0.0411}'
    new_xyz_3 = f'0.075 {handle_y_3} {handle_z - 0.0411}'
    new_rpy = '0 0 0'

    new_xyz_mesh_right = f'0 0 2'
    new_xyz_mesh_left12 = f'0.01 {-0.015 * scale} 2'
    # print(new_xyz_mesh_left12)
    new_rpy_mesh_visual = '1.57 0 0'
    new_rpy_mesh_collision = '0 0 0'

    new_xyz_right = f'-0.015 {0.406 * scale} 0.0411'
    new_xyz_left1 = f'-0.01 {-0.2635 * scale} 0.'
    new_xyz_left2 = f'-0.025 {-0.1425 * scale} 0.0411'

    new_scale = f"1. 1. {scale} "
    # ルート要素として<robot>タグを作成
    robot = ET.Element("robot", name="closet")

    for link in root.findall("link"):
        if link.get("name") == "closet_box":
            for visual in link.findall(".//visual"):
                # print("きたーーーーーーーーーー")
                mesh = visual.find(".//mesh")
                if mesh is not None:
                    # print("メッシ")
                    mesh.set("scale", new_scale)

            for collision in link.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is not None:
                    # print("messonn")
                    mesh.set("scale", new_scale)
                    mesh.set("xyz", '0 0 0')
    
        elif link.get("name") == "door_right":
            for visual in link.findall(".//visual"):
                # print(visual)
                mesh = visual.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                if origin is not None:
                    origin.set("xyz", new_xyz_mesh_right) 


            for collision in link.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                # print("originきた")
                if origin is not None:
                    origin.set("xyz",  new_xyz_mesh_right)
            
        elif link.get("name") == "door_left" or link.get("name") == "door_left2":
            for visual in link.findall(".//visual"):
                mesh = visual.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                if origin is not None:
                    # print("origin_set")
                    origin.set("xyz", new_xyz_mesh_left12) 


            for collision in link.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                if origin is not None:
                    origin.set("xyz",  new_xyz_mesh_left12)
        

    for joint in root.findall('joint'):
        if joint.get('name') == "handle_joint1":
            origin = joint.find('origin')
            if origin is not None:
                # print("ペッツ")
                origin.set('xyz', new_xyz_1)
                # origin.set('rpy', new_rpy)
        elif joint.get('name') == "handle_joint2":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_2)
                # origin.set('rpy', new_rpy)
        elif joint.get('name') == "handle_joint3":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_3)
                # origin.set('rpy', new_rpy)

            
        elif joint.get('name') == "door_right_joint":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_right)
        elif joint.get('name') == "door_left_joint":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_left1)
        elif joint.get('name') == "door_left2_joint":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_left2)

    ##### closet 内部の構造の決定
    divided_number = random.randint(2, 3)
    # divided_number = 2
    
    for partation_number in range(divided_number - 1):
        partation_position = devided_2_position(scale_id, divided_number, partation_number)
        name = f"partation_{partation_number}"
        link_path = "partation"
        # new_link = setting_link(name, scale_number, link_path, partation_position)
        setting_link(root, scale_id, name, 10, link_path, partation_position, False)
        setting_joint(root, f"partation_{partation_number}_joint",  f"partation_{partation_number}")

    for partation_number in range(divided_number):
        structure_number = random.randint(0, 1)


        # structure_number = 1


        if structure_number == 0: #### pole
            pole_number = random.randint(1, 2)

            # pole_number = 1


            if pole_number == 1:
                position = [0, 0, 0]
                position[0] = random.uniform(-0.31, -0.26)
                position[1] = detect_pole_boaed_y(scale, partation_number, divided_number)
                position[2] = random.uniform(1.5, 1.7)
                ### poleの設定
                setting_link(root, scale_id, f"pole_{partation_number}", 10, f"pole_{divided_number}_{partation_number}", position, True, 1.0, "1.57 0. 0.")
                setting_joint(root, f"pole_{partation_number}_joint",  f"pole_{partation_number}")
                ####### 服の設定#########################################################################
                # print()
                max = closet_y[1] * scale_id / divided_number / 0.03
                object_number = random.randint(3, int(max))
                margin = (0.855 * scale_id / divided_number) -(0.03 * object_number)
                margin_list = []
                object_list = np.random.randint(0, 2, size=(object_number))
                for j in range(object_number):
                    margin_j = random.uniform(0, margin)
                    margin_list.append(margin_j)
                    margin -= margin_j
                    position_object = position
                    # position_object[0] = -0.21
                    # position_object[2] += 0.015
                    base_position = cal_base_position(scale_id, divided_number, partation_number, closet_y)
                    position_object[1] = base_position + sum(margin_list) + 0.03 * j + 0.015

                    if object_list[j] == 0:
                        setting_link(root, scale_id, f"object_{partation_number}_{j}", 10, "cloth", position, False)
                    elif object_list[j] == 1:
                        setting_link(root, scale_id, f"object_{partation_number}_{j}", 10, "coat", position, False)

                    setting_joint(root, f"object_{partation_number}_{j}_joint",  f"object_{partation_number}_{j}")

            else:
                position_1 = [0, 0, 0]
                position_2 = [0, 0, 0]
                position_1[0], position_2[0] = np.random.uniform(-0.23, -0.19, 2)
                y_max = random.uniform(1.5, 1.7)
                position_1[1] = detect_pole_boaed_y(scale, partation_number, divided_number)
                position_2[1] = detect_pole_boaed_y(scale, partation_number, divided_number)
                position_1[2] = y_max
                position_2[2] = y_max/2

                position_list = [position_1, position_2]
                max = 0.855 * scale_id / divided_number / 0.03
                # print(f"max ============ {max}")
                for height_number, position in enumerate(position_list):
                    ###pole の設定
                    setting_link(root, scale_id, f"pole_{partation_number}_{height_number}", 10, f"pole_{divided_number}_{partation_number}", position_list[height_number], True, 1.0, "1.57 0. 0.")
                    setting_joint(root, f"pole_{partation_number}_{height_number}_joint",  f"pole_{partation_number}_{height_number}")
                    object_number = random.randint(3, int(max))
                    margin = (0.855 *scale_id / divided_number) -(0.03 * object_number)
                    margin_list = []
                    for j in range(object_number):
                        margin_j = random.uniform(0, margin)
                        margin_list.append(margin_j)
                        margin -= margin_j
                        position_object = position
                        position_object[0] = -0.21
                        # position_object[2] += 0.015
                        base_position = cal_base_position(scale_id, divided_number, partation_number, closet_y)
                        position_object[1] = base_position + sum(margin_list) + 0.03 * j + 0.015

                        setting_link(root, scale_id, f"object_{partation_number}_{height_number}_{j}", 10, "cloth", position, False)
                        setting_joint(root, f"object_{partation_number}_{height_number}_{j}_joint",  f"object_{partation_number}_{height_number}_{j}")
        
        elif structure_number == 1: #### bord
            board_number = random.randint(2, 4)

            # board_number = 4

            success = True
            z_list = setting_zlist(board_number)
            position = [0, 0, 0]
            position[1] = detect_pole_boaed_y(scale, partation_number, divided_number)
            # print(f"{divided_number} :::: {z_list}")
            for i in range(len(z_list)):
                if i > 0:
                    position[2] += z_list[i-1]
                    setting_link(root, scale_id, f"board_{partation_number}_{i}", 10, f"board_{divided_number}_{partation_number}", position, True, 1.0, "1.57 0. 0.")
                    setting_joint(root, f"board_{partation_number}_{i}_joint",  f"board_{partation_number}_{i}")
                ### 物の設定
                object_list = []
                for name in object_name:
                    object_height = object_data[name][0]
                    if object_height < (z_list[i] - 0.04):
                        object_list.append(name)
                success = True
                # print(object_list)
                while success:
                    object_number =  random.randint(1, len(object_list))
                    object_id = random.sample(range(0, len(object_list)), object_number)
                    object = []
                    for o in object_id:
                        object.append(object_list[o])
                    scale_list = []
                    width_list = []
                    for name in object:
                        base_scale = object_data[name][2]
                        height_scale = z_list[i] / object_data[name][0]
                        scale_list.append(min([base_scale, height_scale]))
                        width_list.append(object_data[name][1])

                    success_2 = True
                    repeat_number = 0
                    while success_2:
                        adapt_width_list = []
                        adapt_scale_list = []
                        for j in range(len(object)):
                            adapt_scale_list.append(random.uniform(1., scale_list[j]))
                            adapt_width_list.append(width_list[j] * adapt_scale_list[j])
                        if sum(adapt_width_list) < 0.855 * scale_id/divided_number:
                            success = False
                            success_2 = False
                        repeat_number += 1
                        if repeat_number == 200:
                            success_2 = False
                margin = ( 0.855 * scale_id/divided_number) - sum(adapt_width_list) - 0.03
                margin_list = setting_zlist(object_number, margin.item())
                for j in range(object_number):
                    margin_j = margin_list[j]
                    position_object = [0, 0, 0]
                    position_object[0] = random.uniform(-0.23, -0.19)
                    position_object[2] = position[2] + 0.015
                    if i == 0:
                        position_object[2] = position[2] + 0.03
                    base_position = cal_base_position(scale_id, divided_number, partation_number, closet_y)
                    position_object[1] = base_position + sum(margin_list) + sum(adapt_width_list[:j]) + adapt_width_list[j]/2

                    if object[j] == "suitscase" or object[j] == "shelve"or object[j] == "box":
                        rand = False
                    else:
                        rand = True

                    setting_link(root, scale_id, f"object_{partation_number}_{i}_{j}", 10, f"{object[j]}", position_object, False, adapt_scale_list[j], rot = "0. 0. 0.", random_rot = rand)
                    setting_joint(root, f"object_{partation_number}_{i}_{j}_joint",  f"object_{partation_number}_{i}_{j}")

    rough_string = ET.tostring(root, encoding="unicode")

    # minidomで整形
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # 不要な空行を削除して保存
    output_file = f'/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets/urdf/closet_for_real/urdf_env/{asset_number}.urdf'
    with open(output_file, "w") as f:
        f.write("\n".join(line for line in pretty_xml.splitlines() if line.strip()))

    # tree.write(f'/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets/urdf/closet_for_real/urdf_real/{asset_number}.urdf')

def edit_urdf_structure(scale_id, asset_number, handle_z_number, door_wide_base = 0.282):
    closet_y = [0.7695, 0.855, 1.026, 1.197, 1.368]
    scale = [9, 10, 12, 14, 16]

    ####### handle の高さの設定
    scale = scale_id
    handle_z = handle_z_number

    tree = ET.parse(f'/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets/urdf/closet_for_real/closet_various_handleposition/closet_10.urdf')
    root = tree.getroot()

    handle_y_1 = -1 * ((door_wide_base * scale  / 2) + 0.0075 * scale)
    handle_y_2 = -1 * ((door_wide_base * scale  / 2) - 0.02 * scale)
    handle_y_3 = -1 * ((door_wide_base * scale  / 2))
    
    new_xyz_1 = f'0.075 {handle_y_1+ 0.07} {handle_z - 0.0411}'
    new_xyz_2 = f'0.075 {handle_y_2} {handle_z - 0.0411}'
    new_xyz_3 = f'0.075 {handle_y_3} {handle_z - 0.0411}'
    new_rpy = '0 0 0'

    new_xyz_mesh_right = f'0 0 2'
    new_xyz_mesh_left12 = f'0.01 {-0.015 * scale} 2'
    # print(new_xyz_mesh_left12)
    new_rpy_mesh_visual = '1.57 0 0'
    new_rpy_mesh_collision = '0 0 0'

    new_xyz_right = f'-0.015 {0.406 * scale} 0.0411'
    new_xyz_left1 = f'-0.01 {-0.2635 * scale} 0.'
    new_xyz_left2 = f'-0.025 {-0.1425 * scale} 0.0411'

    new_scale = f"1. 1. {scale} "
    # ルート要素として<robot>タグを作成
    robot = ET.Element("robot", name="closet")

    for link in root.findall("link"):
        if link.get("name") == "closet_box":
            for visual in link.findall(".//visual"):
                # print("きたーーーーーーーーーー")
                mesh = visual.find(".//mesh")
                if mesh is not None:
                    # print("メッシ")
                    mesh.set("scale", new_scale)

            for collision in link.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is not None:
                    print("messonn")
                    mesh.set("scale", new_scale)
    
        elif link.get("name") == "door_right":
            for visual in link.findall(".//visual"):
                # print(visual)
                mesh = visual.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                if origin is not None:
                    origin.set("xyz", new_xyz_mesh_right) 


            for collision in link.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                # print("originきた")
                if origin is not None:
                    origin.set("xyz",  new_xyz_mesh_right)
            
        elif link.get("name") == "door_left" or link.get("name") == "door_left2":
            for visual in link.findall(".//visual"):
                mesh = visual.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                if origin is not None:
                    print("origin_set")
                    origin.set("xyz", new_xyz_mesh_left12) 


            for collision in link.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                if origin is not None:
                    origin.set("xyz",  new_xyz_mesh_left12)
        

    for joint in root.findall('joint'):
        if joint.get('name') == "handle_joint1":
            origin = joint.find('origin')
            if origin is not None:
                # print("ペッツ")
                origin.set('xyz', new_xyz_1)
                # origin.set('rpy', new_rpy)
        elif joint.get('name') == "handle_joint2":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_2)
                # origin.set('rpy', new_rpy)
        elif joint.get('name') == "handle_joint3":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_3)
                # origin.set('rpy', new_rpy)

            
        elif joint.get('name') == "door_right_joint":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_right)
        elif joint.get('name') == "door_left_joint":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_left1)
        elif joint.get('name') == "door_left2_joint":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_left2)

    ##### closet 内部の構造の決定
    divided_number = random.randint(2, 3)
    
    for partation_number in range(divided_number - 1):
        partation_position = devided_2_position(scale_id, divided_number, partation_number)
        name = f"partation_{partation_number}"
        link_path = "partation"
        # new_link = setting_link(name, scale_number, link_path, partation_position)
        setting_link(root, scale_id, name, 10, link_path, partation_position, False)
        setting_joint(root, f"partation_{partation_number}_joint",  f"partation_{partation_number}")

    for partation_number in range(divided_number):
        structure_number = random.randint(0, 1)

        if structure_number == 0: #### pole
            pole_number = random.randint(1, 2)

            if pole_number == 1:
                position = [0, 0, 0]
                position[0] = random.uniform(-0.31, -0.26)
                position[1] = detect_pole_boaed_y(scale, partation_number, divided_number)
                position[2] = random.uniform(1.5, 1.7)
                ### poleの設定
                setting_link(root, scale_id, f"pole_{partation_number}", 10, f"pole_{divided_number}_{partation_number}", position, True, 1.0, "1.57 0. 0.")
                setting_joint(root, f"pole_{partation_number}_joint",  f"pole_{partation_number}")

            else:
                position_1 = [0, 0, 0]
                position_2 = [0, 0, 0]
                position_1[0], position_2[0] = np.random.uniform(-0.23, -0.19, 2)
                position_1[1] = detect_pole_boaed_y(scale, partation_number, divided_number)
                position_2[1] = detect_pole_boaed_y(scale, partation_number, divided_number)
                y_max = random.uniform(1.5, 1.7)
                position_1[2] = y_max
                position_2[2] = y_max/2

                position_list = [position_1, position_2]
                # max = closet_y[scale_id] / divided_number / 0.03
                # print(f"max ============ {max}")
                for height_number, position in enumerate(position_list):
                    ###pole の設定
                    setting_link(root, scale_id, f"pole_{partation_number}_{height_number}", 10, f"pole_{divided_number}_{partation_number}", position_list[height_number], True, 1.0, "1.57 0. 0.")
                    setting_joint(root, f"pole_{partation_number}_{height_number}_joint",  f"pole_{partation_number}_{height_number}")
        
        elif structure_number == 1: #### bord
            board_number = random.randint(2, 4)
            success = True
            z_list = setting_zlist(board_number)
            position = [0, 0, 0]
            position[1] = detect_pole_boaed_y(scale, partation_number, divided_number)
            # print(f"{divided_number} :::: {z_list}")
            for i in range(len(z_list)):
                if i > 0:
                    position[2] += z_list[i-1]
                    setting_link(root, scale_id, f"board_{partation_number}_{i}", 10, f"board_{divided_number}_{partation_number}", position, True, 1.0, "1.57 0. 0.")
                    setting_joint(root, f"board_{partation_number}_{i}_joint",  f"board_{partation_number}_{i}")

    rough_string = ET.tostring(root, encoding="unicode")

    # print(rough_string)

    # minidomで整形
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # 不要な空行を削除して保存
    output_file = f'/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets/urdf/closet_for_real/urdf_env/{asset_number}.urdf'
    with open(output_file, "w") as f:
        f.write("\n".join(line for line in pretty_xml.splitlines() if line.strip()))

def edit_urdf_handle_position(scale_id, asset_number, handle_z_number, door_wide_base = 0.282):
    closet_y = [0.7695, 0.855, 1.026, 1.197, 1.368]
    scale = [9, 10, 12, 14, 16]

    ####### handle の高さの設定
    # scale_number = scale[scale_id]
    scale = scale_id
    handle_z = handle_z_number

    tree = ET.parse(f'/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets/urdf/closet_for_real/closet_various_handleposition/closet_{10}.urdf')
    root = tree.getroot()

    # handle_y_1 = -1 * ((door_wide_base * scale  / 2) + 0.0075 * scale) + 0.075
    handle_y_1 = -1 * ((door_wide_base * scale  / 2) + 0.0075 * scale)
    handle_y_2 = -1 * ((door_wide_base * scale  / 2) - 0.02 * scale)
    handle_y_3 = -1 * ((door_wide_base * scale  / 2))
    
    new_xyz_1 = f'0.075 {handle_y_1+ 0.07} {handle_z - 0.0411}'
    new_xyz_2 = f'0.075 {handle_y_2} {handle_z - 0.0411}'
    new_xyz_3 = f'0.075 {handle_y_3} {handle_z - 0.0411}'
    new_rpy = '0 0 0'

    new_xyz_mesh_right = f'0 0 2'
    new_xyz_mesh_left12 = f'0.01 {-0.015 * scale} 2'
    # print(new_xyz_mesh_left12)
    new_rpy_mesh_visual = '1.57 0 0'
    new_rpy_mesh_collision = '0 0 0'

    new_xyz_right = f'-0.01 {0.406 * scale} 0.0411'
    new_xyz_left1 = f'-0.01 {-0.2635 * scale} 0.'
    new_xyz_left2 = f'-0.025 {-0.1425 * scale} 0.0411'

    new_scale = f"1. 1. {scale} "
    new_scale_handle = f"1. 0.98 {scale} "


    # ルート要素として<robot>タグを作成
    robot = ET.Element("robot", name="closet")

    for link in root.findall("link"):
        if link.get("name") == "closet_box":
            for visual in link.findall(".//visual"):
                # print("きたーーーーーーーーーー")
                mesh = visual.find(".//mesh")
                if mesh is not None:
                    # print("メッシ")
                    mesh.set("scale", new_scale)

            for collision in link.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is not None:
                    print("messonn")
                    mesh.set("scale", new_scale)
    
        elif link.get("name") == "door_right":
            for visual in link.findall(".//visual"):
                # print(visual)
                mesh = visual.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale_handle)
                origin = collision.find("origin")
                if origin is not None:
                    origin.set("xyz", new_xyz_mesh_right) 


            for collision in link.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                # print("originきた")
                if origin is not None:
                    origin.set("xyz",  new_xyz_mesh_right)
            
        elif link.get("name") == "door_left" or link.get("name") == "door_left2":
            for visual in link.findall(".//visual"):
                mesh = visual.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale_handle)
                origin = collision.find("origin")
                if origin is not None:
                    print("origin_set")
                    origin.set("xyz", new_xyz_mesh_left12) 


            for collision in link.findall(".//collision"):
                mesh = collision.find(".//mesh")
                if mesh is not None:
                    mesh.set("scale", new_scale)
                origin = collision.find("origin")
                if origin is not None:
                    origin.set("xyz",  new_xyz_mesh_left12)
        

    for joint in root.findall('joint'):
        if joint.get('name') == "handle_joint1":
            origin = joint.find('origin')
            if origin is not None:
                print("ペッツ")
                origin.set('xyz', new_xyz_1)
                # origin.set('rpy', new_rpy)
        elif joint.get('name') == "handle_joint2":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_2)
                # origin.set('rpy', new_rpy)
        elif joint.get('name') == "handle_joint3":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_3)
                # origin.set('rpy', new_rpy)

            
        elif joint.get('name') == "door_right_joint":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_right)
        elif joint.get('name') == "door_left_joint":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_left1)
        elif joint.get('name') == "door_left2_joint":
            origin = joint.find('origin')
            if origin is not None:
                origin.set('xyz', new_xyz_left2)

    # 不要な空行を削除して保存
    tree.write(f'/home/engawa/py_ws/visual_servo/src/network/predict_opening_door/dataset/predict_opening_wall/assets/urdf/closet_for_real/urdf_env/{asset_number}.urdf')

def detect_pole_boaed_y(scale, partation_number, devided_number):
    door_wide = 0.855 * scale
    if devided_number == 2:
        if partation_number == 0:
            y = -door_wide / 4
        elif partation_number == 1:
            y = door_wide / 4
    if devided_number == 3:
        if partation_number == 0:
            y = -door_wide / 3
        elif partation_number == 1:
            y = 0
        elif partation_number == 2:
            y = door_wide / 3

    return y

def devided_2_position(scale_id, divided_number, partation_number):
    if divided_number == 2 and partation_number == 0:
        position = 0.
    
    if divided_number == 3:
       position =  0.855 * scale_id / 6
    #    p
       if partation_number == 0:
        position *= -1

    return [0, position, 0]

def setting_zlist(divided_number, total_number = 1.74):
    # print(total_number)
    random_numbers = np.random.dirichlet(np.ones(divided_number) * 8)
    # 合計を指定の値にスケーリング
    random_numbers *= total_number
    
    return random_numbers

def setting_link(root, scale_id, name, scale_number, link_path, position, pole_board, scale = 1.0, rot = "0. 0. 0.", random_rot = False):
    if pole_board:
        file_name = "dae"
        scale = f"1 1 {scale_id}"
    else:
        file_name = "stl"
        scale = random.uniform(1., scale)
        scale = f"{scale} {scale} {scale}"

    if random_rot:
        rot_z = random.uniform(-0.2, 0.2)
        rot = f"0. 0. {rot_z}"

    
    
    scale_number = float(scale_number)/10.
    new_link = ET.SubElement(root, "link", {"name": f"{name}"})
    visual = ET.SubElement(new_link, "visual")
    visual_origin = ET.SubElement(visual, "origin", {
            "xyz": f"{position[0]} {position[1]} {position[2]}",  # x, y, z 位置
            "rpy": rot   # roll, pitch, yaw 回転
                })
    visual_geometry = ET.SubElement(visual, "geometry")
    visual_mesh = ET.SubElement(visual_geometry, "mesh", {
            "filename": f"package://closet_for_real/meshes/{scale_number}/visual/{link_path}.{file_name}",
            "scale": f"{scale}"})
    
    collision = ET.SubElement(new_link, "collision")
    collision_origin = ET.SubElement(collision, "origin", {
            "xyz": "0.0 0.0 5.0",  # x, y, z 位置
            "rpy": "0.0 0.0 0.0"   # roll, pitch, yaw 回転
                })
    collision_geometry = ET.SubElement(collision, "geometry")
    collision_mesh = ET.SubElement(collision_geometry, "mesh", {
            "filename": "package://closet_for_real/meshes/collision.stl",
            "scale": "1.0 1.0 1.0"})


def setting_joint(root, name, child):
    new_joint = ET.SubElement(root, "joint", {"name": f"{name}", "type":"fixed"})
    ET.SubElement(new_joint, "parent", {"link": "closet_box"})
    ET.SubElement(new_joint, "child", {"link": child})
    ET.SubElement(new_joint, "origin", {"xyz": "0. 0. 0.", "rpy": "0. 0. 0."})


def cal_base_position(scale_id, devided_number, partation_number, closet_y):
    if devided_number == 2:
        if partation_number == 0:
            base_position = - 0.855 * scale_id/2
        elif partation_number == 1:
            base_position = 0
    elif devided_number == 3:
        if partation_number == 0:
            base_position = - 0.855 * scale_id/2
        elif partation_number == 1:
            base_position = - 0.855 * scale_id/6
        elif partation_number == 2:
            base_position = + 0.855 * scale_id/6
    
    return base_position



def prettify(element):
    """ElementTreeを受け取り、整形されたXML文字列を返す"""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


if __name__ == '__main__':
    edit_urdf_real(0, 1, 1127)





