import torch

x = torch.rand(4, 4)
y0 = torch.rand(5)
y1 = torch.rand(4)
z0 = torch.rand(3, 2, 5)
z1 = torch.rand(3, 5, 4)
w = torch.rand(2, 3, 4, 5)
r0 = torch.rand(2, 5)
r1 = torch.rand(3, 5, 4)
r2 = torch.rand(2, 4)
s0 = torch.rand(2, 3, 5, 7)
s1 = torch.rand(11, 3, 17, 5)


# identity
a0 = torch.einsum('i', y0)
a1 = torch.einsum('ij', x)
a2 = torch.einsum('ijk', z0)

a0_ = y0
a1_ = x
a2_ = z0
assert torch.allclose(a0, a0_)
assert torch.allclose(a1, a1_)
assert torch.allclose(a2, a2_)

# permute
b0 = torch.einsum('ij->ji', x)
b1 = torch.einsum('ba', x)
b2 = torch.einsum('jki', z0)
b3 = torch.einsum('ijk->kij', z0)
b4 = torch.einsum('kjil', w)
b5 = torch.einsum('...ij->...ji', w)
b6 = torch.einsum('abc...->cba...', w)

b0_ = x.permute(1, 0)
b1_ = x.permute(1, 0)
b2_ = z0.permute(2, 0, 1)
b3_ = z0.permute(2, 0, 1)
b4_ = w.permute(2, 1, 0, 3)
b5_ = w.transpose(-2, -1)
b6_ = w.transpose(0, 2)


assert torch.allclose(b0, b0_)
assert torch.allclose(b1, b1_)
assert torch.allclose(b2, b2_)
assert torch.allclose(b3, b3_)
assert torch.allclose(b4, b4_)
assert torch.allclose(b5, b5_)
assert torch.allclose(b6, b6_)


# trace
c = torch.einsum('ii', x)
c_ = torch.sum(torch.diag(x))
assert torch.allclose(c, c_)

# sum
d0 = torch.einsum('ij->', x)
d1 = torch.einsum('xyz->', z0)
d2 = torch.einsum('ijkl->', w)

d0_ = torch.sum(x)
d1_ = torch.sum(z0)
d2_ = torch.sum(w)
assert torch.allclose(d0, d0_)
assert torch.allclose(d1, d1_)
assert torch.allclose(d2, d2_)

# sum axis
e0 = torch.einsum('ijk->i', z0)
e1 = torch.einsum('ijk->j', z0)
e2 = torch.einsum('ijk->ij', z0)

e0_ = torch.sum(z0, dim=[1,2])
e1_ = torch.sum(z0, dim=[0,2])
e2_ = torch.sum(z0, dim=2)
assert torch.allclose(e0, e0_)
assert torch.allclose(e1, e1_)
assert torch.allclose(e2, e2_)


# matrix-vector
f0 = torch.einsum('ij,j->i', r0, y0)
f1 = torch.einsum('i,jki->jk', y1, r1)

f0_ = r0 @ y0
f1_ = torch.sum(y1 * r1, dim=-1)
assert torch.allclose(f0, f0_)
assert torch.allclose(f1, f1_)

# vector-vector outer product
g0 = torch.einsum('i,j->ij', y0, y1)
g1 = torch.einsum('a,b,c,d->abcd', y0, y1, y0, y1)

g0_ = torch.outer(y0, y1)
t = torch.outer(y0, y1)
t = torch.stack([t for _ in range(len(y0))], dim=-1) * y0
g1_ = torch.stack([t for _ in range(len(y1))], dim=-1) * y1
assert torch.allclose(g0, g0_)
assert torch.allclose(g1, g1_)



# batch mm
h0 = torch.einsum('bij,bjk->bik', z0, z1)
h1 = torch.einsum('bjk,bij->bik', z1, z0)
h0_ = torch.bmm(z0, z1)
h1_ = torch.bmm(torch.transpose(z1, 1,2), torch.transpose(z0, 1,2))
h1_ = torch.transpose(h1_, 1,2)
assert torch.allclose(h0, h0_)
assert torch.allclose(h1, h1_)

# bilinear
i = torch.einsum('bn,anm,bm->ba', r0, r1, r2)
t = torch.tensordot(r0, r1, ([1], [1]))
i_ = torch.sum(t * torch.stack([r2 for _ in range(t.shape[1])], dim=1), dim=-1)
assert torch.allclose(i, i_)

# tensor contraction
j = torch.einsum('pqrs,tqvr->pstv', s0, s1)
j_ = torch.tensordot(s0, s1, ([1, 2], [1, 3]))
assert torch.allclose(j, j_)