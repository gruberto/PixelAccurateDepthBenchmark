class Dataset:

    def __init__(self, data_root):
        self.data_root = data_root

    def get_clear_sequence(self, scene, daytime):
        if scene == 'all':
            scenes = ['scene1', 'scene2', 'scene3', 'scene4']
        else:
            scenes = [scene]

        samples = []
        for scene in scenes:
            for i in range(10):
                samples.append('{}_{}_{}_{}'.format(scene, daytime, 'clear', i))

        return samples

    def get_fog_sequence(self, scene, daytime, visibility):
        if scene == 'all':
            scenes = ['scene1', 'scene2', 'scene3', 'scene4']
        else:
            scenes = [scene]

        samples = []
        for scene in scenes:
            for i in range(10):
                samples.append('{}_{}_{}_{}'.format(scene, daytime, 'fog{}'.format(visibility), i))

        return samples

    def get_rain_sequence(self, scene, daytime, rainfall_rate):
        if scene == 'all':
            scenes = ['scene1', 'scene2', 'scene3', 'scene4']
        else:
            scenes = [scene]

        samples = []
        for scene in scenes:
            for i in range(10):
                if rainfall_rate == 0:
                    samples.append('{}_{}_{}_{}'.format(scene, daytime, 'clear', i))
                else:
                    samples.append('{}_{}_{}_{}'.format(scene, daytime, 'rain{}'.format(rainfall_rate), i))

        return samples