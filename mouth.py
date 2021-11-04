def get_mouth_landmarks(face):
    return [face.landmark[84], face.landmark[85], face.landmark[86], face.landmark[38], face.landmark[72],
            face.landmark[81], face.landmark[82], face.landmark[87], face.landmark[37], face.landmark[0],
            face.landmark[11], face.landmark[12], face.landmark[13], face.landmark[14], face.landmark[178],
            face.landmark[179], face.landmark[180], face.landmark[181], face.landmark[88], face.landmark[89],
            face.landmark[90], face.landmark[91], face.landmark[95], face.landmark[96], face.landmark[183],
            face.landmark[77], face.landmark[42], face.landmark[78], face.landmark[146], face.landmark[184],
            face.landmark[62], face.landmark[191], face.landmark[74], face.landmark[80], face.landmark[41],
            face.landmark[15], face.landmark[16], face.landmark[315], face.landmark[316], face.landmark[317],
            face.landmark[402], face.landmark[403], face.landmark[404], face.landmark[405], face.landmark[320],
            face.landmark[321], face.landmark[375], face.landmark[307], face.landmark[325], face.landmark[291],
            face.landmark[292], face.landmark[306], face.landmark[308], face.landmark[408], face.landmark[409],
            face.landmark[270], face.landmark[407], face.landmark[415], face.landmark[304], face.landmark[269],
            face.landmark[267], face.landmark[302], face.landmark[39], face.landmark[40], face.landmark[185],
            face.landmark[17], face.landmark[314], face.landmark[319], face.landmark[272], face.landmark[310],
            face.landmark[318], face.landmark[324], face.landmark[271], face.landmark[311], face.landmark[303],
            face.landmark[268], face.landmark[312], face.landmark[73], face.landmark[61], face.landmark[76]]


def get_inner_mouth(face):
    return [face.landmark[13], face.landmark[312], face.landmark[311], face.landmark[310], face.landmark[415],
            face.landmark[308], face.landmark[324], face.landmark[318], face.landmark[402], face.landmark[317],
            face.landmark[14], face.landmark[87], face.landmark[178], face.landmark[88], face.landmark[95],
            face.landmark[78], face.landmark[191], face.landmark[80], face.landmark[81], face.landmark[82]]


def get_outer_mouth(face):
    return [face.landmark[0], face.landmark[267], face.landmark[269], face.landmark[270], face.landmark[409],
            face.landmark[291], face.landmark[375], face.landmark[321], face.landmark[405], face.landmark[314],
            face.landmark[17], face.landmark[84], face.landmark[181], face.landmark[91], face.landmark[146],
            face.landmark[61], face.landmark[185], face.landmark[40], face.landmark[39], face.landmark[37]]


def get_mouth_mask(face):
    return get_outer_mouth(face) + get_inner_mouth(face)
