import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import GAN

def interpolate_latent_space(generator, latent_size, num_steps=10, device='cuda'):
    # 生成两个随机潜在向量
    z1 = torch.randn(1, latent_size, 1, 1, device=device)
    z2 = torch.randn(1, latent_size, 1, 1, device=device)

    # 在两个潜在向量之间进行线性插值
    alpha_values = torch.linspace(0, 1, num_steps, device=device)
    interpolated_z = z1 * (1 - alpha_values.view(-1, 1, 1, 1)) + z2 * alpha_values.view(-1, 1, 1, 1)

    # 将生成器置于评估模式，并生成插值图像
    generator.eval()
    with torch.no_grad():
        interpolated_images = generator(interpolated_z)

    # 保存插值图像
    save_image(interpolated_images, '/home/zhangtq21/ANN/HW4/codes/results/interpolation.png', nrow=num_steps, normalize=True)


generator = GAN.Generator(1, 100, 100).to('cuda')
checkpoint = torch.load('/home/zhangtq21/ANN/HW4/codes/results/latent-100_hidden-100_batch-64_num-train-steps-5000/4999/generator.bin', map_location=torch.device('cuda'))
generator.load_state_dict(checkpoint)
generator.eval()
latent_size = 100
interpolate_latent_space(generator, latent_size)
