import torch
import os
import tqdm
from toolbox.conversions import connectivity_to_dgl
import dgl

class Base_Generator(torch.utils.data.Dataset):
    def __init__(self, name, path_dataset, num_examples):
        self.path_dataset = path_dataset
        self.name = name
        self.num_examples = num_examples
    
    def converting_dataset_to_dgl(self, l_data, **kwargs):
        print("Converting data to DGL format")
        l_data_dgl = []
        for data,target in tqdm.tqdm(l_data):
            elt_dgl = connectivity_to_dgl(data, **kwargs)
            elt_dgl = dgl.add_self_loop(elt_dgl)
            l_data_dgl.append((elt_dgl,target))
        print("Conversion ended.")
        return l_data_dgl

    def load_dataset(self, use_dgl=False):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        filename_dgl = self.name + '_dgl.pkl'
        path = os.path.join(self.path_dataset, filename)
        path_dgl = os.path.join(self.path_dataset, filename_dgl)
        data_exists = os.path.exists(path)
        data_dgl_exists = os.path.exists(path_dgl)
        if data_exists or data_dgl_exists:
            if use_dgl:
                if data_dgl_exists:
                    print('Reading dataset at {}'.format(path_dgl))
                    data = torch.load(path_dgl)
                else:
                    print('DGL file does not exist. Reading from regular file.')
                    print('Reading dataset at {}'.format(path))
                    data = torch.load(path)
                    data = self.converting_dataset_to_dgl(data)
                    print('Saving dataset at {}'.format(path_dgl))
                    torch.save(data, path_dgl)
            else:
                print('Reading dataset at {}'.format(path))
                data = torch.load(path)
            self.data = list(data)
        else:
            print('Creating dataset at {}'.format(path))
            l_data = self.create_dataset()
            print('Saving dataset at {}'.format(path))
            torch.save(l_data, path)
            print('Creating dataset at {}'.format(path_dgl))
            l_data_dgl = self.converting_dataset_to_dgl(l_data)
            print('Saving dataset at {}'.format(path_dgl))
            torch.save(l_data_dgl, path_dgl)
            if use_dgl:
                self.data = l_data_dgl
            else:
                self.data = l_data
    
    @staticmethod
    def _solution_conversion(target, dgl_graph):
        """This function should convert a tensor target into a dgl graph, it should be overwritten in your inherited generator class
        Depending on the task at hand, our method is for the graph to have various features (if the problm can be used for both, we add both):
          - edge classification : put the solution in graph.edata['solution']
          - node classification : put the solution in graph.ndata['solution']
        Obviously, this is our way of handling the data, it is a compromise between space and readability, you can change it as you see fit, as long as your pytorch_lightning module handles it.
        """
        raise NotImplementedError("This function should be implemented in your generator. 'Base_Generator' is an abstract class and this function should be overwritten in your inherited generator class.")
    
    def remove_files(self):
        base_file = os.path.join(self.path_dataset, self.name + '.pkl')
        dgl_file = os.path.join(self.path_dataset, self.name + '_dgl.pkl')
        try:
            os.remove(base_file)
        except FileNotFoundError:
            print(f'File {base_file} already removed.')
        try:
            os.remove(dgl_file)
        except FileNotFoundError:
            print(f'File {dgl_file} already removed.')
    
    def create_dataset(self):
        l_data = []
        for _ in tqdm.tqdm(range(self.num_examples)):
            example = self.compute_example()
            l_data.append(example)
        return l_data

    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)