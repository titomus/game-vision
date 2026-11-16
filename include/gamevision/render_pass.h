#pragma once
// GameVision SDK: host-driven D3D11 post-processing. Does not inject into games.
// The caller supplies GPU render resources it legitimately owns or receives from a supported engine.
#include <d3d11.h>
#include <memory>

namespace gamevision {
struct RenderOptions {
    // 0 = passthrough, 1 = subtle, 2 = balanced, 3 = strong
    // 4 = visualize depth, 5 = visualize motion magnitude (debug only).
    int preset = 2;
    float intensity = 1.0f;
    bool hasDepth = false;
    bool hasMotion = false;
    // For the diagnostics pass, depth and motion must be readable normalized float SRVs.
    // Motion expects XY screen-UV displacement, not pixels or world coordinates.
};

// The input/output textures must have compatible dimensions and sample count 1.
// Do not use the same texture for input SRV and output RTV: SRV/RTV aliasing is undefined here.
// Caller must restore any D3D11 pipeline state it wants to preserve after Render.
// Render performs no CPU readback and does not own the host swapchain.
class RenderPass final {
public:
    explicit RenderPass(ID3D11Device* device);
    ~RenderPass();
    RenderPass(const RenderPass&) = delete;
    RenderPass& operator=(const RenderPass&) = delete;
    void Render(ID3D11DeviceContext* context,
                ID3D11ShaderResourceView* color,
                ID3D11RenderTargetView* destination,
                unsigned width, unsigned height,
                const RenderOptions& options = {},
                ID3D11ShaderResourceView* depth = nullptr,
                ID3D11ShaderResourceView* motion = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
