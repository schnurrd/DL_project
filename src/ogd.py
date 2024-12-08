import torch


class OrthogonalGradientDescent:
    """
    Implements Orthogonal Gradient Descent (OGD) with the Ground Truth Logit (GTL) variant.

    After finishing training on each task (with your standard training objective, typically cross-entropy),
    call `update_basis(dataloader)` to store gradient directions that preserve the ground-truth logit of old tasks.
    This ensures that subsequent gradient updates do not reduce performance on previously learned tasks.

    Usage:
        1. Train the model on a new task normally using your standard loss.
        2. After the task is done, call `ogd.update_basis(old_task_dataloader)` to store constraints.
        3. For future training steps, call `ogd.step()` after computing gradients and before the optimizer step
            to project new gradients onto the complement of previously stored directions.
    """

    def __init__(self, model, optimizer, device="cpu", max_basis_size=200, reduce_basis=False):
        """
        Args:
            model (nn.Module): The model being optimized. Must have a `model.ogd_basis` attribute,
                                or this class will create it.
            optimizer (torch.optim.Optimizer): The underlying optimizer (e.g. SGD, Adam).
            device (str): Device to perform computation on ('cpu', 'cuda', or 'mps').
            max_basis_size (int): Maximum number of gradient directions to store.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.max_basis_size = max_basis_size

        if not hasattr(self.model, "ogd_basis"):
            self.model.ogd_basis = torch.empty((0, 0), device=self.device)

        self.current_dim = self._get_param_dim()
        self.reduce_basis = reduce_basis

    def update_basis(self, dataloader):
        """
        Update the orthogonal gradient basis using OGD-GTL on given dataloader samples.

        OGD-GTL variant: We focus on the ground truth logit for each sample.
        We do not use the original training loss here. Instead, we:
            - Extract the ground-truth logit per sample
            - Define a loss = -sum(ground_truth_logits), which encourages gradients
                that preserve or increase those logits.

        After backprop, we store and orthonormalize these gradient directions, adding them
        to the ogd_basis. Future gradients will be projected onto the complement of these directions.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the current (old) task.
        """

        # Before collecting new gradients, ensure dimensions match
        self._ensure_correct_dimension()

        sampled_gradients = []
        num_samples = len(dataloader)

        # Limit the number of gradients to self.max_basis_size
        num_to_sample = min(self.max_basis_size, num_samples)
        sampled_indices = torch.randperm(num_samples)[:num_to_sample]

        for idx, (inputs, labels) in enumerate(dataloader):  # might make sense to batch this
            if idx not in sampled_indices:
                continue
            inputs, labels = inputs.to(self.device), torch.tensor(
                labels, device=self.device
            )

            # Forward pass and compute gradient for each sample
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # Extract ground truth logits
            gt_logits = outputs.gather(
                1, labels.view(-1, 1)
            ).squeeze()  # shape: (batch_size,)

            # OGD-GTL loss: negative sum to get a direction that would increase gt logits
            loss = -gt_logits.sum()
            loss.backward()

            # Convert gradients to a vector
            grad_vec = self._parameters_to_grad_vector(
                self.model.parameters()
            ).unsqueeze(
                1
            )  # Column vector
            sampled_gradients.append(grad_vec)

        if not sampled_gradients:
            self.optimizer.zero_grad()
            return

        # Combine the sampled gradients and orthonormalize
        new_basis = torch.cat(sampled_gradients, dim=1)
        if self.model.ogd_basis.numel() > 0:
            combined_basis = torch.cat([self.model.ogd_basis.T, new_basis], dim=1)
            orthonormal_basis = self._orthonormalize(combined_basis)
        else:
            orthonormal_basis = self._orthonormalize(new_basis)


        # Truncate the basis to retain only the most informative vectors by norm
        if self.reduce_basis and orthonormal_basis.shape[1] > self.max_basis_size:
            orthonormal_basis = self._truncate_basis_by_norm(orthonormal_basis)

        # Update the model's basis
        self.model.ogd_basis = orthonormal_basis.T

        self.optimizer.zero_grad()

    def step(self):
        """
        Perform an OGD step after computing gradients for the new task and before calling `optimizer.step()`.

        This method projects the current gradients onto the orthogonal complement of previously stored directions,
        ensuring no interference with old tasks. Then it updates the model parameters using the corrected gradients.
        """

        self._ensure_correct_dimension()

        # Convert current gradients to a vector
        grad_vec = self._parameters_to_grad_vector(self.model.parameters())

        # Project gradients onto the orthogonal basis
        if self.model.ogd_basis.numel() == 0:
            corrected_grad = grad_vec
        else:
            proj_grad = self._project_vec(grad_vec, self.model.ogd_basis.T)
            corrected_grad = grad_vec - proj_grad

        # Replace gradients in the model with the corrected gradients
        pointer = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                num_params = param.numel()
                param.grad.copy_(
                    corrected_grad[pointer : pointer + num_params].view(param.size())
                )
                pointer += num_params

        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    # Utility methods for parameter management
    def _parameters_to_grad_vector(self, parameters):
        """
        Convert parameter gradients into a single flattened vector.
        """
        return torch.cat([p.grad.view(-1) for p in parameters if p.grad is not None])

    def _orthonormalize(self, basis):
        """
        Orthonormalize the basis using Gram-Schmidt.
        """
        if self.device.type == "mps":
            q, _ = torch.linalg.qr(basis.to("cpu"))
            return q.to(self.device)
        q, _ = torch.linalg.qr(basis)
        return q

    def _project_vec(self, vector, basis):
        """
        Project vector onto the given basis.

        Args:
            vector (torch.Tensor): A [dim] vector representing current gradients.
            basis (torch.Tensor): A [dim, n_vectors] orthonormal basis matrix.

        Returns:
            torch.Tensor: The projection of `vector` onto the subspace spanned by `basis`.
        """
        if basis.size(1) == 0:
            return torch.zeros_like(vector)
        return basis @ (basis.T @ vector)

    def _truncate_basis_by_norm(self, basis):
        """
        Truncate the basis to the top `max_basis_size` vectors by their norm.

        Args:
            basis (torch.Tensor): [dim, n_vectors] orthonormalized basis.

        Returns:
            torch.Tensor: Truncated basis of size [dim, max_basis_size].
        """
        # Compute the norms of each column vector in the basis
        norms = torch.norm(basis, dim=0)  # Norm along the rows (vector norms)
        # Find indices of the top `max_basis_size` vectors by norm
        _, top_indices = torch.topk(norms, self.max_basis_size, sorted=False)
        # Select the corresponding vectors
        return basis[:, top_indices]

    def _get_param_dim(self):
        # Returns the total number of parameters
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _ensure_correct_dimension(self):
        # Check if current parameter dimension matches ogd_basis dimension
        new_dim = self._get_param_dim()
        # If ogd_basis is empty, initialize old_dim to new_dim
        old_dim = (
            self.model.ogd_basis.shape[1]
            if self.model.ogd_basis.numel() > 0
            else new_dim
        )

        if new_dim > old_dim and self.model.ogd_basis.numel() > 0:
            # Zero-pad the old basis to match the new dimension
            old_basis = self.model.ogd_basis
            n_vectors = old_basis.shape[0]
            padded_basis = torch.zeros(
                (n_vectors, new_dim), device=self.device, dtype=old_basis.dtype
            )
            new_pointer = 0
            old_pointer = 0
            for name, param in self.model.named_parameters():
                new_size = param.numel()
                old_size = self.model.old_param_size_map[name]
                padded_basis[:, new_pointer:new_pointer+old_size] = old_basis[:, old_pointer : old_pointer + old_size]
                new_pointer += new_size
                old_pointer += old_size

            self.model.ogd_basis = padded_basis

        # Update current_dim
        self.current_dim = new_dim
