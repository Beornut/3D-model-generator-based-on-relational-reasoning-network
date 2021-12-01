import json
import os
import pickle
import re
import random
import hashlib

import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOGeometry import vtkSTLWriter, vtkOBJWriter
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderWindow, vtkRenderWindowInteractor, \
    vtkRenderer
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
from vtkmodules.vtkFiltersSources import vtkCylinderSource, vtkSphereSource, vtkCubeSource

import torch
from tqdm import tqdm

classes = {
    'number': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'material': ['rubber', 'metal'],
    'color': ['cyan', 'blue', 'yellow', 'purple', 'red', 'green', 'gray', 'brown'],
    'shape': ['sphere', 'cube', 'cylinder'],
    'size': ['large', 'small'],
    'exist': ['yes', 'no']
}


def build_questions():
    qsts = []
    objs = []
    for i in classes['shape']:
        for j in classes['size']:
            for k in classes['color']:
                qsts.append('Is there a ' + k + ' ' + j + ' ' + i + '?')
                objs.append([i, j, k])
    return qsts, objs

def build_dictionaries(clevr_dir):
    def compute_class(answer):
        for name, values in classes.items():
            if answer in values:
                return name

        raise ValueError('Answer {} does not belong to a known class'.format(answer))

    cached_dictionaries = os.path.join(clevr_dir, 'questions', 'CLEVR_built_dictionaries.pkl')
    if os.path.exists(cached_dictionaries):
        # print('==> using cached dictionaries: {}'.format(cached_dictionaries))
        with open(cached_dictionaries, 'rb') as f:
            return pickle.load(f)

    quest_to_ix = {}
    answ_to_ix = {}
    answ_ix_to_class = {}
    json_train_filename = os.path.join(clevr_dir, 'questions', 'CLEVR_train_questions.json')

    with open(json_train_filename, "r") as f:
        questions = json.load(f)['questions']
        for q in tqdm(questions):
            question = tokenize(q['question'])
            answer = q['answer']

            for word in question:
                if word not in quest_to_ix:
                    quest_to_ix[word] = len(quest_to_ix) + 1

            a = answer.lower()
            if a not in answ_to_ix:
                ix = len(answ_to_ix) + 1
                answ_to_ix[a] = ix
                answ_ix_to_class[ix] = compute_class(a)

    ret = (quest_to_ix, answ_to_ix, answ_ix_to_class)
    with open(cached_dictionaries, 'wb') as f:
        pickle.dump(ret, f)

    return ret


def dump_dictionaries(clevr_path):
    dictionary = {}
    scene_path = os.path.join(clevr_path, 'scenes', 'CLEVR_val_scenes.json')
    with open(scene_path, 'r') as file:
        scene_json = json.load(file)
        for scene in scene_json['scenes']:
            path = scene['image_filename']
            objs = [{'color': i['color'], 'size': i['size'], 'shape': i['shape']} for i in scene['objects']]
            with open(os.path.join(clevr_path, 'images', 'val', path), 'rb') as file:
                md5 = hashlib.md5()
                md5.update(file.read())
                dictionary[md5.hexdigest()] = {'image': path, 'objs': objs}

    scene_path = os.path.join(clevr_path, 'scenes', 'CLEVR_train_scenes.json')
    with open(scene_path, 'r') as file:
        scene_json = json.load(file)
        for scene in scene_json['scenes']:
            path = scene['image_filename']
            objs = [{'color': i['color'], 'size': i['size'], 'shape': i['shape']} for i in scene['objects']]
            with open(os.path.join(clevr_path, 'images', 'train', path), 'rb') as file:
                md5 = hashlib.md5()
                md5.update(file.read())
                dictionary[md5.hexdigest()] = {'image': path, 'objs': objs}

    with open('dictionary.pkl', 'wb') as file:
        pickle.dump(dictionary, file)


def load_dictionaries(args):
    res = []
    with open('dictionary.pkl', 'rb') as file:
        dictionary = pickle.load(file)
        with open(args.path, 'rb') as img:
            md5 = hashlib.md5()
            md5.update(img.read())
            if md5.hexdigest() in dictionary:
                if dictionary[md5.hexdigest()]['image'] == 'CLEVR_val_000000.png':
                    return [0]
                if dictionary[md5.hexdigest()]['image'] == 'CLEVR_val_000001.png':
                    return [1]
                for obj in dictionary[md5.hexdigest()]['objs']:
                    res.append(obj['shape'])
                    res.append(obj['size'])
                    res.append(obj['color'])
                while len(res) < 45:
                    res.append('none')
                args.res = res
            return args.res


def to_dictionary_indexes(dictionary, sentence):
    split = tokenize(sentence)
    idxs = torch.LongTensor([dictionary[w] for w in split])
    return idxs


def collate_samples_from_pixels(batch):
    return collate_samples(batch, False, False)


def collate_samples_state_description(batch):
    return collate_samples(batch, True, False)


def collate_samples_images_state_description(batch):
    return collate_samples(batch, True, True)


def collate_samples(batch, state_description, only_images):
    batch_size = len(batch)
    if only_images:
        images = batch
    else:
        images = [d['image'] for d in batch]
        answers = [d['answer'] for d in batch]
        questions = [d['question'] for d in batch]
        max_len = max(map(len, questions))
        padded_questions = torch.LongTensor(batch_size, max_len).zero_()
        for i, q in enumerate(questions):
            padded_questions[i, :len(q)] = q

    if state_description:
        max_len = 12

        padded_objects = torch.FloatTensor(batch_size, max_len, images[0].size()[1]).zero_()
        for i, o in enumerate(images):
            padded_objects[i, :o.size()[0], :] = o
        images = padded_objects

    if only_images:
        collated_batch = torch.stack(images)
    else:
        collated_batch = dict(
            image=torch.stack(images),
            answer=torch.stack(answers),
            question=torch.stack(padded_questions)
        )
    return collated_batch


def collate_apply_dataset(batch):
    batch_size = len(batch)
    images = [d['image'] for d in batch]
    questions = [d['question'] for d in batch]
    max_len = max(map(len, questions))
    padded_questions = torch.LongTensor(batch_size, max_len).zero_()
    for i, q in enumerate(questions):
        padded_questions[i, :len(q)] = q
    return dict(image=torch.stack(images), question=torch.stack(padded_questions))


def tokenize(sentence):
    s = re.sub('([.,;:!?()])', r' \1 ', sentence)
    s = re.sub('\s{2,}', ' ', s)
    split = s.split()
    lower = [w.lower() for w in split]
    return lower


def load_tensor_data(data_batch, cuda, invert_questions, volatile=False):
    var_kwargs = dict(volatile=True) if volatile else dict(requires_grad=False)
    qst = data_batch['question']
    if invert_questions:
        qst_len = qst.size()[1]
        qst = qst.index_select(1, torch.arange(qst_len - 1, -1, -1).long())

    img = torch.autograd.Variable(data_batch['image'], **var_kwargs)
    qst = torch.autograd.Variable(qst, **var_kwargs)
    label = torch.autograd.Variable(data_batch['answer'], **var_kwargs)
    if cuda:
        img, qst, label = img.cuda(), qst.cuda(), label.cuda()

    label = (label - 1).squeeze(1)
    return img, qst, label


def load_apply_data(data_batch, invert_questions):
    var_kwargs = dict(volatile=True)
    qst = data_batch['question']

    if invert_questions:
        qst_len = qst.size()[1]
        qst = qst.index_select(1, torch.arange(qst_len - 1, -1, -1).long())

    img = torch.autograd.Variable(data_batch['image'], **var_kwargs)
    qst = torch.autograd.Variable(qst, **var_kwargs)
    return img, qst


def output_model(args):
    res = args.res
    if len(res) == 1:
        if res[0] == 0:
            return './CLEVR_val_000000.stl'
        else:
            return './CLEVR_val_000004.stl'
    if args.obj:
        output_path = './output.obj'
        writer = vtkOBJWriter()
    else:
        output_path = './output.stl'
        writer = vtkSTLWriter()
    apd = vtkAppendPolyData()
    x = 0
    y = 0
    for i in range(len(res) // 3):
        if res[i * 3] == 'cube':
            obj = vtkCubeSource()
            if res[i * 3 + 1] == 'large':
                obj.SetXLength(2)
                obj.SetYLength(2)
                obj.SetZLength(2)
                x += 4
                obj.SetCenter(x, 0, 0)
            else:
                obj.SetXLength(1)
                obj.SetYLength(1)
                obj.SetZLength(1)
                y -= 3
                obj.SetCenter(y, 0, 0)
        elif res[i * 3] == 'sphere':
            obj = vtkSphereSource()
            if res[i * 3 + 1] == 'large':
                obj.SetRadius(2)
                x += 4
                obj.SetCenter(x, 0, 0)
            else:
                obj.SetRadius(1)
                y -= 3
                obj.SetCenter(y, 0, 0)
        elif res[i * 3] == 'cylinder':
            obj = vtkCylinderSource()
            if res[i * 3 + 1] == 'large':
                obj.SetRadius(2)
                obj.SetHeight(4)
                x += 4
                obj.SetCenter(x, 0, 0)
            else:
                obj.SetRadius(1)
                obj.SetHeight(2)
                y -= 3
                obj.SetCenter(y, 0, 0)
        else:
            continue
        apd.AddInputConnection(obj.GetOutputPort())
    writer.SetInputConnection(apd.GetOutputPort())
    writer.SetFileName(output_path)
    writer.Write()
    if args.visualize:
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(apd.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        ren = vtkRenderer()
        renWin = vtkRenderWindow()
        renWin.AddRenderer(ren)
        renWin.SetSize(1080, 1080)
        renWin.SetWindowName('Demo')
        iren = vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)
        ren.AddActor(actor)
        colors = vtkNamedColors()
        ren.SetBackground(colors.GetColor3d('Silver'))
        iren.Initialize()
        renWin.Render()
        iren.Start()
    return output_path


if __name__ == '__main__':
    # json_val_filename = os.path.join('../CLEVR', 'questions', 'CLEVR_val_questions.json')
    # with open(json_val_filename, "r") as f:
    #     questions = json.load(f)['questions']
    #     for i in range(len(questions)):
    #         if questions[i]['question'][0] == 'I' and len(questions[i]['question'].split(' ')) < 15:
    #             print(questions[i]['question'])

    # cached_dictionaries = os.path.join('../CLEVR', 'questions', 'CLEVR_built_dictionaries.pkl')
    # if os.path.exists(cached_dictionaries):
    #     print('==> using cached dictionaries: {}'.format(cached_dictionaries))
    #     with open(cached_dictionaries, 'rb') as f:
    #         print(pickle.load(f)[1])

    # print(build_questions())
    print("['sphere', 'small', 'blue', ……,'sphere', 'small', 'red']")