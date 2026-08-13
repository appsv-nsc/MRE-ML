# Save the paths of dataset dirs.
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'base_mre_map_2d':
            return 'E:/data/mre'
        elif dataset == 'example2':
            return 'E:/data/example2'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError