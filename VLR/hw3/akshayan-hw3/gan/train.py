from glob import glob
import os
import torch
from utils import get_fid, interpolate_latent_space, save_plot
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import VisionDataset

amp_enabled = True

def build_transforms():
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.
    ds_transforms = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                             std=[0.5, 0.5, 0.5])])
    return ds_transforms

# subtratcts a const value at a fixed time step
def adjust_lr(optimizer, curr_step, initial_lr, total_steps, name):
    for groups in optimizer.param_groups:
        if groups['lr'] > 0:
            groups['lr'] -= initial_lr / total_steps

        #print(curr_step, name, groups['lr'])


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K steps.
    # The learning rate for the generator should be decayed to 0 over 100K steps.

    optim_discriminator = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0, 0.9))
    #scheduler_discriminator = torch.optim.lr_scheduler.StepLR(optim_discriminator, 2, 0.5)
    scheduler_discriminator = adjust_lr
    optim_generator = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0, 0.9))
    scheduler_generator =  adjust_lr #torch.optim.lr_scheduler.StepLR(optim_generator, 2, 0.1)
    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
):
    torch.backends.cudnn.benchmark = True
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    disc_loss_total = 0
    gen_loss_total = 0
    disc_logs = 0
    gen_logs = 0
    iters = 0
    fids_list = []
    iters_list = []
    while iters < num_iterations:
        for train_batch in train_loader:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                train_batch = train_batch.cuda()
                # TODO 1.2: compute generator outputs and discriminator outputs
                # 1. Compute generator output -> the number of samples must match the batch size.
                # 2. Compute discriminator output on the train batch.
                # 3. Compute the discriminator output on the generated data.
                discrim_real = disc(train_batch)
                fake_data = gen(train_batch.shape[0])
                discrim_fake = disc(fake_data)

                # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.
                # To compute interpolated data, draw eps ~ Uniform(0, 1)
                # interpolated data = eps * fake_data + (1-eps) * real_data
                eps = torch.rand((train_batch.shape[0], 1, 1, 1)).cuda()
                eps = eps.repeat(1, fake_data.shape[1], fake_data.shape[2], fake_data.shape[3])
                interp = eps * fake_data + (1 - eps) * train_batch
                discrim_interp = disc(interp)

                discriminator_loss = disc_loss_fn(
                    discrim_real, discrim_fake, discrim_interp, interp, lamb
                )
                disc_loss_total += discriminator_loss.item()
                disc_logs += 1
            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward(retain_graph=True)
            scaler.step(optim_discriminator)

            if iters % 100 == 0 and iters != 0:
                scheduler_discriminator(optim_discriminator, iters, 0.0002, 5000 , "disc")

            if iters % 5 == 0:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    # TODO 1.2: Compute samples and evaluate under discriminator.
                    fake_data = gen(batch_size)
                    discrim_fake = disc(fake_data)
                    generator_loss = gen_loss_fn(discrim_fake)
                    print(iters, generator_loss.item())
                    gen_loss_total += generator_loss.item()
                    gen_logs += 1

                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                if iters % 100 == 0 and iters != 0:
                    print("Iter: [{}/{}], Discriminator loss: {}, Generator loss: {}".format(
                    iters, 30000, disc_loss_total / disc_logs, gen_loss_total / gen_logs))
                    scheduler_generator(optim_generator, iters, 0.0002, 1000, "gen")
                    disc_loss_total = 0
                    gen_loss_total = 0
                    disc_logs = 0
                    gen_logs = 0

            if iters % log_period == 0: # and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].
                        temp_transform = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
                        generated_samples = gen(batch_size)
                        generated_samples = temp_transform(generated_samples)

                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    #torch.jit.save(gen, prefix + "/generator.pt")
                    #torch.jit.save(disc, prefix + "/discriminator.pt")
                    # torch.save(gen, prefix + "/generator.pt")
                    # torch.save(disc, prefix + "/discriminator.pt")
                    torch.cuda.empty_cache()
                    os.environ["PYTORCH_JIT"] = "1"
                    fid = fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=16,
                        num_gen= 10_000,
                    )
                    print(f"Iteration {iters} FID: {fid}")
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
    torch.cuda.empty_cache()
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=16,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")
