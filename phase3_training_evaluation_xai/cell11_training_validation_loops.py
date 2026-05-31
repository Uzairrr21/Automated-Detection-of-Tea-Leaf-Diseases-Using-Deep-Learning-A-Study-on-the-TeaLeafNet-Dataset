# ==============================================================
# CELL 11 — Training & Validation Loops
# ==============================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, use_amp,
                    use_mixup=False, mixup_alpha=0.2):
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), \
                       labels.to(DEVICE, non_blocking=True)

        # ── Softer MixUp to improve train accuracy ──
        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(
                imgs, labels, alpha=mixup_alpha
            )

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                out  = model(imgs)
                loss = (
                    mixup_criterion(criterion, out, y_a, y_b, lam)
                    if use_mixup else criterion(out, labels)
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            # ── Slightly relaxed clipping ──
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

        else:
            out  = model(imgs)

            loss = (
                mixup_criterion(criterion, out, y_a, y_b, lam)
                if use_mixup else criterion(out, labels)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )

            optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        # ── Better MixUp accuracy estimation ──
        preds = out.argmax(1)

        if use_mixup:
            correct += (
                lam * (preds == y_a).sum().item()
                + (1 - lam) * (preds == y_b).sum().item()
            )
        else:
            correct += (preds == labels).sum().item()

        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, use_amp):
    model.eval()

    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), \
                       labels.to(DEVICE, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
        else:
            out  = model(imgs)
            loss = criterion(out, labels)

        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


def run_phase(model, train_loader, val_loader,
              lr, epochs, phase_name, history,
              use_mixup=False, mixup_alpha=0.2,
              weight_decay=1e-4, warmup_epochs=1,
              optimizer_type="adamw"):

    USE_AMP = (DEVICE.type == "cuda")

    if optimizer_type == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)

        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Reduced label smoothing to reduce underfitting ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_acc, best_state = 0.0, None

    print(f"\n{'='*65}")
    print(f"  {phase_name}")
    print(f"  Optimizer={optimizer_type.upper()} | LR={lr} | Epochs={epochs}")
    print(f"  MixUp={use_mixup} α={mixup_alpha} | WD={weight_decay} | Warmup={warmup_epochs}")
    print(f"  GradClip=1.0 | LabelSmoothing=0.08 | AMP={USE_AMP}")
    print(f"{'='*65}")

    for ep in range(1, epochs + 1):

        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            USE_AMP,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha
        )

        vl_loss, vl_acc = validate(
            model,
            val_loader,
            criterion,
            USE_AMP
        )

        scheduler.step()

        history["phase"].append(phase_name)
        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_acc:
            best_acc   = vl_acc
            best_state = copy.deepcopy(model.state_dict())

        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Ep {ep:03d}/{epochs} | "
            f"Tr Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
            f"Vl Loss={vl_loss:.4f} Acc={vl_acc:.4f} | "
            f"LR={cur_lr:.2e} | {time.time()-t0:.1f}s"
        )

    print(f"\n✅ {phase_name} done — Best Val Acc = {best_acc:.4f}")

    model.load_state_dict(best_state)

    return model