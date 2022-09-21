from heartBeatsExtraction import get_train_test_heart_beats
import pandas as pd

if __name__ == '__main__':
    size = 5
    train_hb, train_cls, test_hb, test_cls = get_train_test_heart_beats(size)

    list_container = []
    for comp in range(150):
        comp_list = []
        for hb in train_hb:
            comp_list.append(hb[comp])
        list_container.append(comp_list)

    comp_str_cont = []
    for num in range(150):
        comp_str_cont.append('Comp' + str(num))

    data = {'Class': train_cls}
    for comp_str, comp_list in zip(comp_str_cont, list_container):
        data[comp_str] = comp_list

    df = pd.DataFrame.from_dict(data)
    shuffled = df.sample(frac=1)
    for row in shuffled:
        print(row)
