from im2mesh.encoder import (
    conv, pix2mesh_cond, pointnet,
    psgn_cond, r2n2, voxels, graphs, pointnet2,
    deepsets, transformer, kpconv, perceiver
)


encoder_dict = {
    'simple_conv': conv.ConvEncoder,
    'resnet18': conv.Resnet18,
    'resnet34': conv.Resnet34,
    'resnet50': conv.Resnet50,
    'resnet101': conv.Resnet101,
    'r2n2_simple': r2n2.SimpleConv,
    'r2n2_resnet': r2n2.Resnet,
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,
    'psgn_cond': psgn_cond.PCGN_Cond,
    'voxel_simple': voxels.VoxelEncoder,
    'pixel2mesh_cond': pix2mesh_cond.Pix2mesh_Cond,
    'dgcnn': graphs.DGCNN,
    'pointnet2': pointnet2.PointNet2MSG,
    'pointnet2ssg': pointnet2.PointNet2SSG,
    'deepsets_dtanh':deepsets.DTanh,
    'deepsets_dtanhx2':deepsets.DTanh2X,
    'transformer': transformer.Pct,
    'kpconv': kpconv.Encoder,
    'perceiver': perceiver.Encoder
}
