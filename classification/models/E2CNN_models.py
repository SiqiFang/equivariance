import torch
import torch.nn as tnn
import torch.nn.functional as F

import escnn.gspaces as gspaces
import escnn.nn as enn

# More information on the E2CNN library can be found here: https://github.com/QUVA-Lab/e2cnn

class BlockRN18ESCNN(enn.EquivariantModule):
    '''
    A single block of the ResNet18 model equivariant to rotations of 90 degrees using the E2CNN library.
    The block consists of two convolutional layers with batch normalization and ReLU activation functions.
    The block also has a skip connection to add the input to the output.

    Args:
    in_type: nn.FieldType
        The input type of the block
    out_type: nn.FieldType
        The output type of the block
    sub_type: dict
        The scalar and vector fields in the block
    stride: int
        The stride of the convolutional layers
    kernel_size: int
        The kernel size of the convolutional layers
    padding: int
        The padding of the convolutional layers
    '''
    def __init__(self, 
                 in_type: enn.FieldType,
                 out_type: enn.FieldType=None,
                 sub_type: enn.FieldType=None,
                 stride=1,
                 kernel_size=3,
                 padding=1,
                ):
        super(BlockRN18ESCNN, self).__init__()

        # if outtype is not specified, it is an intermediate block
        if out_type is None and sub_type is None:
            out_type = in_type

        self.in_type = in_type
        self.out_type = out_type
        self.skipcon = None

        # We use sub_type to specify the scalar and vector fields in the block due to mixed representations for the non-linearity
        if sub_type is not None:

            # The sub_type is a tuple of the scalar and vector fields
            scalar_fields = sub_type.get('scalar', [])
            vector_fields = sub_type.get('vector', [])
            out_type = scalar_fields + vector_fields

            # The non-linearity is a combination of the norm and ReLU
            relu = enn.ReLU(scalar_fields)
            norm_relu = enn.NormNonLinearity(vector_fields)

            self.c1 = enn.R2Conv(in_type = in_type, out_type = out_type, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
            self.n1 = enn.GNormBatchNorm(out_type)
            self.a1 = enn.MultipleModule(out_type, ['relu']*len(scalar_fields) + ['norm']*len(vector_fields), [(relu, 'relu'), (norm_relu, 'norm')])
            self.c2 = enn.R2Conv(in_type = out_type, out_type = out_type, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
            self.n2 = enn.GNormBatchNorm(out_type)
            self.a2 = enn.MultipleModule(out_type, ['relu']*len(scalar_fields) + ['norm']*len(vector_fields), [(relu, 'relu'), (norm_relu, 'norm')])
        else:
            self.c1 = enn.R2Conv(in_type = in_type, out_type = out_type, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
            self.n1 = enn.InnerBatchNorm(out_type)
            self.a1 = enn.ReLU(out_type)
            self.c2 = enn.R2Conv(in_type = out_type, out_type = out_type, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
            self.n2 = enn.InnerBatchNorm(out_type)
            self.a2 = enn.ReLU(out_type)

        if stride != 1 or in_type != out_type:
            self.skipcon = enn.R2Conv(in_type = in_type, out_type = out_type, kernel_size = 1, stride = stride, padding = 0, bias = False)

    def forward(self, x):
        '''
        The forward pass of the block. The input is passed through the convolutional layers and the skip connection is added to the output.

        Args:
        x: GeometricTensor
            The input tensor to the block

        Returns:
        out: GeoemtricTensor
            The output tensor of the block
        '''
        out = self.c1(x)
        out = self.n1(out)
        out = self.a1(out)
        out = self.c2(out)
        out = self.n2(out)

        if self.skipcon is not None:
            out = out + self.skipcon(x)
        else:
            out = out + x
        out = self.a2(out)
        return out

    def evaluate_output_shape(self, input_shape: tuple):
        '''
        Args:
        input_shape: tuple
            The input shape to the block

        Returns:
        output_shape: tuple
            The output shape of the block
        '''
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.skipcon is not None:
            return self.skipcon.evaluate_output_shape(input_shape)
        else:
            return input_shape

class BlockRN18Default(tnn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super(BlockRN18Default, self).__init__()
        self.c1 = tnn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.n1 = tnn.BatchNorm2d(out_channels)
        self.a1 = tnn.ReLU()
        self.c2 = tnn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.n2 = tnn.BatchNorm2d(out_channels)
        self.a2 = tnn.ReLU()
        self.skipcon = None
        if stride != 1 or in_channels != out_channels:
            self.skipcon = tnn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = self.c1(x)
        out = self.n1(out)
        out = self.a1(out)
        out = self.c2(out)
        out = self.n2(out)
        if self.skipcon is not None:
            out = out + self.skipcon(x)
        else:
            out = out + x
        out = self.a2(out)
        return out

class RegRepRN18(tnn.Module):
    '''
    A ResNet18 model equivariant to rotations of 90 degrees using the E2CNN library.
    The model consists of an input block, four layers of blocks, and a linear layer at the end.

    Args:
    n0: int
        The number of channels in the first stage
    n1: int
        Corresponds to the number of classes in the dataset for the linear layer
    kernel_size: int
        The kernel size of the convolutional layers
    img_channels: int
        The number of channels in the input image
    '''

    def __init__(self, n0=32, n1=2, kernel_size=3, img_channels=3):
        super(RegRepRN18, self).__init__()
        self.n0 = n0
        self.n1 = n1
        self.img_channels = img_channels

        # track the layer number
        self.num_layers = 0
             
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2

        # stages is the number of channels in each stage and n is the number of blocks in each stage
        self.stages = [n0, n0 * 2, n0 * 4, n0 * 8]
        self.n = [2, 2, 2, 2]

        # specify the group and representation with the number of channels
        self.r2_act = gspaces.rot2dOnR2(N = 4)

        # specify the input type and store the input type for when we wrap images into a GeometricTensor
        in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
        out_type = enn.FieldType(self.r2_act, n0 * [self.r2_act.regular_repr])
        self.in_type = in_type

        # Track the layer number
        self.num_layers += 1

        # inblock of the ResNet
        self.inblock1 = enn.SequentialModule(
            enn.R2Conv(in_type = in_type, out_type = out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type)
        )

        self._in_type = self.inblock1.out_type
        self.layer2 = self._make_layer(BlockRN18ESCNN, self.stages[0], self.n[0])

        self._in_type = self.layer2.out_type
        self.layer3 = self._make_layer(BlockRN18ESCNN, self.stages[1], self.n[1])

        self._in_type = self.layer3.out_type
        self.layer4 = self._make_layer(BlockRN18ESCNN, self.stages[2], self.n[2])

        # The last layer has the trivial representation as the output
        self._in_type = self.layer4.out_type
        self.layer5 = self._make_layer(BlockRN18ESCNN, self.stages[3], self.n[3], totrivial = True)

        self.linear = tnn.Linear(self.layer5.out_type.size, n1)


    def _make_layer(self, block, out_channels, blocks, totrivial = False) -> enn.SequentialModule:
        '''
        Make a layer of blocks in the ResNet model.

        Args:
        block: nn.Module
            The block to use in the layer
        out_channels: int
            The number of channels in the layer
        blocks: int
            The number of blocks in the layer
        totrivial: bool
            Whether to use the trivial representation in the last block of the layer

        Returns:
            nn.SequentialModule
        '''
        
        # Track the layer number
        self.num_layers += 1

        # List of blocks in the layer
        layers = []

        inner_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])

        # Specify the out_type according to the representation
        if totrivial:
            out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.trivial_repr])
        else:
            out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])

        for i in range(blocks):

            # If it is the last block in the layer, the out_type is the 
            if i == blocks - 1:
                out_type = out_type
            else:
                out_type = inner_type
            layers.append(block(self._in_type, out_type))
            self._in_type = out_type

        return enn.SequentialModule(*layers)    

    def forward(self, x):
        '''
        The forward pass of the model.

        Args:
        x: torch.Tensor
            The input tensor to the model

        Returns:
        x: torch.Tensor
            The output tensor of the model.
        '''
        x = enn.GeometricTensor(x, self.in_type)
        x = self.inblock1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.tensor
        b,c, w, h = x.shape
        x = F.avg_pool2d(x, (w, h))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class TrivialRepRN18(tnn.Module):
    '''
    A ResNet18 model equivariant to rotations of 90 degrees using the E2CNN library.
    The model consists of an input block, four layers of blocks, and a linear layer at the end.

    Args:
    n0: int
        The number of channels in the first stage
    n1: int
        Corresponds to the number of classes in the dataset for the linear layer
    kernel_size: int
        The kernel size of the convolutional layers
    img_channels: int
        The number of channels in the input image
    '''

    def __init__(self, n0=32, n1=2, kernel_size=3, img_channels=3):
        super(TrivialRepRN18, self).__init__()
        self.n0 = n0
        self.n1 = n1
        self.img_channels = img_channels

        # track the layer number
        self.num_layers = 0
             
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2

        # stages is the number of channels in each stage and n is the number of blocks in each stage
        self.stages = [n0, n0 * 2, n0 * 4, n0 * 8]
        self.n = [2, 2, 2, 2]

        # specify the group and representation with the number of channels
        self.r2_act = gspaces.rot2dOnR2(N = 4)

        # specify the input type and store the input type for when we wrap images into a GeometricTensor
        in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
        out_type = enn.FieldType(self.r2_act, n0 * [self.r2_act.trivial_repr])
        self.in_type = in_type

        # Track the layer number
        self.num_layers += 1

        # inblock of the ResNet
        self.inblock1 = enn.SequentialModule(
            enn.R2Conv(in_type = in_type, out_type = out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type)
        )

        self._in_type = self.inblock1.out_type
        self.layer2 = self._make_layer(BlockRN18ESCNN, self.stages[0], self.n[0])

        self._in_type = self.layer2.out_type
        self.layer3 = self._make_layer(BlockRN18ESCNN, self.stages[1], self.n[1])

        self._in_type = self.layer3.out_type
        self.layer4 = self._make_layer(BlockRN18ESCNN, self.stages[2], self.n[2])

        # The last layer has the trivial representation as the output
        self._in_type = self.layer4.out_type
        self.layer5 = self._make_layer(BlockRN18ESCNN, self.stages[3], self.n[3])

        self.linear = tnn.Linear(self.layer5.out_type.size, n1)


    def _make_layer(self, block, out_channels, blocks) -> enn.SequentialModule:
        '''
        Make a layer of blocks in the ResNet model.

        Args:
        block: nn.Module
            The block to use in the layer
        planes: int
            The number of channels in the layer
        blocks: int
            The number of blocks in the layer

        Returns:
            nn.SequentialModule
        '''
        
        # Track the layer number
        self.num_layers += 1

        # List of blocks in the layer
        layers = []

        out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.trivial_repr])

        for i in range(blocks):
            layers.append(block(self._in_type, out_type))
            self._in_type = out_type

        return enn.SequentialModule(*layers)    

    def forward(self, x):
        '''
        The forward pass of the model.

        Args:
        x: torch.Tensor
            The input tensor to the model

        Returns:
        x: torch.Tensor
            The output tensor of the model.
        '''
        x = enn.GeometricTensor(x, self.in_type)
        x = self.inblock1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.tensor
        b,c, w, h = x.shape
        x = F.avg_pool2d(x, (w, h))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
class TrivialIrrRepRN18(tnn.Module):
    '''
    A ResNet18 model equivariant to rotations of 90 degrees using the E2CNN library.
    The model consists of an input block, four layers of blocks, and a linear layer at the end.

    Args:
    n0: int
        The number of channels in the first stage
    n1: int
        Corresponds to the number of classes in the dataset for the linear layer
    kernel_size: int
        The kernel size of the convolutional layers
    img_channels: int
        The number of channels in the input image
    '''

    def __init__(self, n0=32, n1=2, kernel_size=3, img_channels=3):
        super(TrivialIrrRepRN18, self).__init__()
        self.n0 = n0
        self.n1 = n1
        self.img_channels = img_channels

        # track the layer number
        self.num_layers = 0
             
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2

        # stages is the number of channels in each stage and n is the number of blocks in each stage
        self.stages = [n0, n0 * 2, n0 * 4, n0 * 8]
        self.n = [2, 2, 2, 2]

        # specify the group and representation with the number of channels
        self.r2_act = gspaces.rot2dOnR2(N = 4)

        # specify the input type and store the input type for when we wrap images into a GeometricTensor
        in_type = enn.FieldType(self.r2_act, img_channels * [self.r2_act.trivial_repr])
        in_scalar = enn.FieldType(self.r2_act, n0//2 * [self.r2_act.trivial_repr])
        in_vector = enn.FieldType(self.r2_act, n0//2 * [self.r2_act.irrep(1)])
        out_type = in_scalar + in_vector
        self.in_type = in_type

        # Track the layer number
        self.num_layers += 1

        # inblock of the ResNet
        self.inblock1 = enn.SequentialModule(
            enn.R2Conv(in_type = in_type, out_type = out_type, kernel_size = kernel_size, stride = 1, padding = padding, bias = False),
            enn.GNormBatchNorm(out_type),
            enn.MultipleModule(out_type, ['relu']*len(in_scalar) + ['norm']*len(in_vector), [(enn.ReLU(in_scalar), 'relu'), (enn.NormNonLinearity(in_vector), 'norm')])
        )

        self._in_type = self.inblock1.out_type
        self.layer2 = self._make_layer(BlockRN18ESCNN, self.stages[0], self.n[0])

        self._in_type = self.layer2.out_type
        self.layer3 = self._make_layer(BlockRN18ESCNN, self.stages[1], self.n[1])

        self._in_type = self.layer3.out_type
        self.layer4 = self._make_layer(BlockRN18ESCNN, self.stages[2], self.n[2])

        # The last layer has the trivial representation as the output
        self._in_type = self.layer4.out_type
        self.layer5 = self._make_layer(BlockRN18ESCNN, self.stages[3], self.n[3], totrivial=True)

        self.linear = tnn.Linear(self.layer5.out_type.size, n1)


    def _make_layer(self, block, out_channels, blocks, totrivial = False) -> enn.SequentialModule:
        '''
        Make a layer of blocks in the ResNet model.

        Args:
        block: nn.Module
            The block to use in the layer
        planes: int
            The number of channels in the layer
        blocks: int
            The number of blocks in the layer

        Returns:
            nn.SequentialModule
        '''

        # Track the layer number
        self.num_layers += 1

        # List of blocks in the layer
        layers = []

        scalar_subtype = enn.FieldType(self.r2_act, out_channels//2 * [self.r2_act.trivial_repr])
        vector_subtype = enn.FieldType(self.r2_act, out_channels//2 * [self.r2_act.irrep(1)])
        sub_type = {'scalar': scalar_subtype, 'vector': vector_subtype}
        inner_type = scalar_subtype + vector_subtype

        # Specify the out_type according to the representation
        if totrivial:
            out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.trivial_repr])
        else:
            out_type = inner_type

        for i in range(blocks):

            # If it is the last block in the layer, the out_type is the 
            if i == blocks - 1:
                out_type = out_type
            else:
                out_type = inner_type
            layers.append(block(self._in_type, out_type, sub_type = sub_type))
            self._in_type = out_type

        return enn.SequentialModule(*layers)    

    def forward(self, x):
        '''
        The forward pass of the model.

        Args:
        x: torch.Tensor
            The input tensor to the model

        Returns:
        x: torch.Tensor
            The output tensor of the model.
        '''
        x = enn.GeometricTensor(x, self.in_type)
        x = self.inblock1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.tensor
        b,c, w, h = x.shape
        x = F.avg_pool2d(x, (w, h))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class DefaultRN18(tnn.Module):
    def __init__(self, n0=32, n1=2, kernel_size=3, img_channels=3):
        super(DefaultRN18, self).__init__()
        self.n0 = n0
        self.n1 = n1
        self.img_channels = img_channels

        # track the layer number
        self.num_layers = 0
             
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2

        # stages is the number of channels in each stage and n is the number of blocks in each stage
        self.stages = [n0, n0 * 2, n0 * 4, n0 * 8]
        self.n = [2, 2, 2, 2]
        
        # Track the layer number
        self.num_layers += 1

        # inblock of the ResNet
        self.inblock1 = BlockRN18Default(img_channels, n0, kernel_size=kernel_size, padding=padding)

        self._in_channels = n0
        self.layer2 = self._make_layer(BlockRN18Default, self.stages[0], self.n[0])

        self._in_channels = self.stages[0]
        self.layer3 = self._make_layer(BlockRN18Default, self.stages[1], self.n[1])

        self._in_channels = self.stages[1]
        self.layer4 = self._make_layer(BlockRN18Default, self.stages[2], self.n[2])

        self._in_channels = self.stages[2]
        self.layer5 = self._make_layer(BlockRN18Default, self.stages[3], self.n[3])

        self.linear = tnn.Linear(self.stages[3], n1)

    def _make_layer(self, block, out_channels, blocks) -> tnn.Sequential:
        '''
        Make a layer of blocks in the ResNet model.

        Args:
        block: nn.Module
            The block to use in the layer
        planes: int
            The number of channels in the layer
        blocks: int
            The number of blocks in the layer
        
        Returns:
            nn.Sequential
        '''

        # Track the layer number
        self.num_layers += 1

        # List of blocks in the layer
        layers = tnn.ModuleList()

        for i in range(blocks):
            layers.append(block(self._in_channels, out_channels))
            self._in_channels = out_channels

        return tnn.Sequential(*layers)

    def forward(self, x):
        '''
        The forward pass of the model.

        Args:
        x: torch.Tensor
            The input tensor to the model

        Returns:
        x: torch.Tensor
            The output tensor of the model.
        '''
        x = self.inblock1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        b,c, w, h = x.shape
        x = F.avg_pool2d(x, (w, h))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x