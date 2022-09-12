def get_classes(annot):
    classes = {}
    cnt = 0
    for i in range(len(annot)):
        if annot[str(i)]["annotation"] not in classes:
            classes[annot[str(i)]["annotation"]] = cnt
            cnt += 1
