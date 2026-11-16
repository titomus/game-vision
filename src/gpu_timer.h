#include <stdexcept>
#pragma once
#include <d3d11.h>
#include <wrl/client.h>
#include <array>
#include <cstddef>
#include <cstdint>

class GpuTimer {
    using Ptr = Microsoft::WRL::ComPtr<ID3D11Query>;
    struct Frame {
        Ptr disjoint, begin, end;
        bool pending=false;
        uint64_t serial=0;
    };
    std::array<Frame, 12> frames{};
    std::size_t writeIndex=0;
    uint64_t nextSerial=1;
    float lastGpuMs=-1.0f;
    uint64_t completed=0;
    uint64_t invalid=0;
public:
    explicit GpuTimer(ID3D11Device* dev) {
        D3D11_QUERY_DESC q{};
        for (auto& frame:frames) {
            q.Query=D3D11_QUERY_TIMESTAMP_DISJOINT;
            if (FAILED(dev->CreateQuery(&q,&frame.disjoint))) throw std::runtime_error("GPU disjoint query failed");
            q.Query=D3D11_QUERY_TIMESTAMP;
            if (FAILED(dev->CreateQuery(&q,&frame.begin))) throw std::runtime_error("GPU timestamp query failed");
            if (FAILED(dev->CreateQuery(&q,&frame.end))) throw std::runtime_error("GPU timestamp query failed");
        }
    }
    bool Begin(ID3D11DeviceContext* ctx) {
        auto& frame=frames[writeIndex];
        // Never overwrite a query still waiting on the GPU.
        if (frame.pending) { ++invalid; return false; }
        ctx->Begin(frame.disjoint.Get());
        ctx->End(frame.begin.Get());
        frame.pending=true;
        frame.serial=nextSerial++;
        return true;
    }
    void End(ID3D11DeviceContext* ctx) {
        auto& frame=frames[writeIndex];
        ctx->End(frame.end.Get());
        ctx->End(frame.disjoint.Get());
        writeIndex=(writeIndex+1)%frames.size();
    }
    void Poll(ID3D11DeviceContext* ctx) {
        // D3D11_ASYNC_GETDATA_DONOTFLUSH ensures this never blocks waiting for GPU work.
        for (auto& frame:frames) {
            if (!frame.pending) continue;
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT meta{};
            HRESULT a=ctx->GetData(frame.disjoint.Get(),&meta,sizeof(meta),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            if (a==S_FALSE) continue;
            uint64_t first=0,last=0;
            HRESULT b=ctx->GetData(frame.begin.Get(),&first,sizeof(first),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            HRESULT c=ctx->GetData(frame.end.Get(),&last,sizeof(last),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            if (b==S_FALSE || c==S_FALSE) continue;
            if (a==S_OK && b==S_OK && c==S_OK && !meta.Disjoint && meta.Frequency>0 && last>=first) {
                lastGpuMs=static_cast<float>((last-first)*1000.0/static_cast<double>(meta.Frequency));
                ++completed;
            } else {
                ++invalid;
            }
            frame.pending=false;
        }
    }
    float LatestMs() const { return lastGpuMs; }
    uint64_t Completed() const {return completed;}
    uint64_t Dropped() const {return invalid;}
};
