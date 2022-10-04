import torchio

class augmentation:

    def __init__(self, cfg) -> None:

            # augmentation
            self.augment = cfg['DATASET']['AUGMENTATION']['augment']
            if self.augment == True:
                self.z_deg = cfg['DATASET']['AUGMENTATION']['z_deg']
                self.y_deg = cfg['DATASET']['AUGMENTATION']['y_deg']
                self.x_deg = cfg['DATASET']['AUGMENTATION']['x_deg']

    def transforms(self):
        
        degree = (self.z_deg, self.y_deg, self.x_deg)
        # translate = (10,10,10)
        transform_rotate = torchio.RandomAffine(degrees=degree, 
        #                                         translation=translate, 
                                                image_interpolation="bspline")
        transform_flip = torchio.RandomFlip(axes=('ap',))
        all_transform = torchio.Compose([transform_rotate,
                                        transform_flip])

        return all_transform