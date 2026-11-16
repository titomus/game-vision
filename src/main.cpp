#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <memory>
#include <fstream>
#include <iomanip>
#include "gpu_timer.h"
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <stdexcept>
#include <string>
using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;

static const char* kShader = R"HLSL(
Texture2D frameTex : register(t0);
SamplerState samp : register(s0);
cbuffer Controls : register(b0) {
    float2 texel;
    float power;
    float split;
    float4 settings;
};
struct VOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
VOut VS(uint id : SV_VertexID) {
    VOut o;
    float2 p = float2((id << 1) & 2, id & 2);
    o.pos = float4(p * float2(2,-2) + float2(-1,1), 0, 1);
    o.uv = p;
    return o;
}
float4 PS(VOut i) : SV_TARGET {
    float2 uv = i.uv;
    float3 c = frameTex.Sample(samp, uv).rgb;
    if (settings.x > 0.5 && !(split > 0.5 && uv.x < 0.5)) {
        float3 n = frameTex.Sample(samp, uv + float2(0,-texel.y)).rgb;
        float3 s = frameTex.Sample(samp, uv + float2(0, texel.y)).rgb;
        float3 e = frameTex.Sample(samp, uv + float2( texel.x,0)).rgb;
        float3 w = frameTex.Sample(samp, uv + float2(-texel.x,0)).rgb;
        float3 lo = min(c, min(min(n,s),min(e,w)));
        float3 hi = max(c, max(max(n,s),max(e,w)));
        float contrast = 1 + power * (settings.x > 2.5 ? .13 : settings.x > 1.5 ? .06 : .025);
        float saturation = 1 + power * (settings.x > 2.5 ? .14 : settings.x > 1.5 ? .05 : .015);
        float edge = saturate(dot(hi - lo, float3(.2126,.7152,.0722)) * 3);
        float sharpen = power * (settings.x > 2.5 ? .48 : settings.x > 1.5 ? .32 : .16);
        c = clamp(c + (c - .25*(n+s+e+w)) * sharpen * (.45+.55*edge), lo, hi);
        c = saturate((c-.5)*contrast+.5);
        float lum = dot(c, float3(.2126,.7152,.0722));
        c = saturate(lerp(lum.xxx,c,saturation));
    }
    if (split > .5 && abs(uv.x-.5) < texel.x*2) c=float3(.99,.80,.24);
    return float4(c,1);
}
)HLSL";

static void Check(HRESULT hr, const char* where) {
    if (FAILED(hr)) {
        char msg[160]{};
        std::snprintf(msg,sizeof(msg),"%s: HRESULT 0x%08lX", where, (unsigned long)hr);
        throw std::runtime_error(msg);
    }
}
struct Controls {
    float texel[2] = {1.0f/1920,1.0f/1080};
    float power = 1, split = 1;
    float settings[4] = {2,0,0,0};
};
static_assert(sizeof(Controls)%16==0,"constant buffer alignment");

class GameVision {
public:
    int preset=2;
    float power=1;
    bool split=true, vsync=false;
    void Initialize(HWND window, unsigned monitor) {
        hwnd=window;
        ComPtr<IDXGIFactory1> factory;
        Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"DXGI factory");
        unsigned found=0;
        bool selected=false;
        for (UINT a=0; !selected; ++a) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapters1(a,&candidate)==DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 info{};
            Check(candidate->GetDesc1(&info),"Adapter info");
            if (info.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            for (UINT o=0;;++o) {
                ComPtr<IDXGIOutput> output;
                if (candidate->EnumOutputs(o,&output)==DXGI_ERROR_NOT_FOUND) break;
                if (found++ != monitor) continue;
                adapter=candidate;
                Check(output.As(&output1),"DXGI output");
                selected=true;
                break;
            }
        }
        if (!selected) throw std::runtime_error("Monitor not found. Try --monitor 0 or 1.");
        D3D_FEATURE_LEVEL levels[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL level{};
        HRESULT hr=D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,levels,2,D3D11_SDK_VERSION,
            &device,&level,&context);
        if (hr==E_INVALIDARG)
            hr=D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,&levels[1],1,D3D11_SDK_VERSION,
                &device,&level,&context);
        Check(hr,"D3D11 device");
        ComPtr<IDXGIDevice> dxgi;
        Check(device.As(&dxgi),"IDXGIDevice");
        ComPtr<IDXGIAdapter> actualAdapter;
        Check(dxgi->GetAdapter(&actualAdapter),"GetAdapter");
        ComPtr<IDXGIFactory2> factory2;
        Check(actualAdapter->GetParent(IID_PPV_ARGS(&factory2)),"DXGI factory2");
        DXGI_SWAP_CHAIN_DESC1 sd{};
        sd.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
        sd.SampleDesc.Count=1;
        sd.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount=2;
        sd.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.Scaling=DXGI_SCALING_STRETCH;
        Check(factory2->CreateSwapChainForHwnd(device.Get(),hwnd,&sd,nullptr,nullptr,&swap),
              "CreateSwapChainForHwnd");
        factory2->MakeWindowAssociation(hwnd,DXGI_MWA_NO_ALT_ENTER);
        auto compile=[](const char* entry,const char* target) {
            ComPtr<ID3DBlob> blob,errors;
            HRESULT result=D3DCompile(kShader,std::strlen(kShader),"gamevision.hlsl",
                nullptr,nullptr,entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&errors);
            if (FAILED(result))
                throw std::runtime_error(errors?
                    std::string((char*)errors->GetBufferPointer(),errors->GetBufferSize()):
                    "HLSL compilation failed");
            return blob;
        };
        auto v=compile("VS","vs_5_0"),p=compile("PS","ps_5_0");
        Check(device->CreateVertexShader(v->GetBufferPointer(),v->GetBufferSize(),nullptr,&vs),"VS");
        Check(device->CreatePixelShader(p->GetBufferPointer(),p->GetBufferSize(),nullptr,&ps),"PS");
        D3D11_SAMPLER_DESC sm{};
        sm.Filter=D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sm.AddressU=sm.AddressV=sm.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;
        sm.MaxLOD=D3D11_FLOAT32_MAX;
        Check(device->CreateSamplerState(&sm,&sampler),"sampler");
        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth=sizeof(Controls);
        bd.Usage=D3D11_USAGE_DEFAULT;
        bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        Check(device->CreateBuffer(&bd,nullptr,&uniforms),"controls");
        gpuTimer=std::make_unique<GpuTimer>(device.Get());
        Resize();
        ResetCapture();
    }
    void OpenCsv(const std::wstring& path) {
        csv.open(path, std::ios::out | std::ios::trunc);
        if (!csv) throw std::runtime_error("Cannot open benchmark CSV file.");
        csv << "elapsed_s,capture_fps,gpu_shader_ms,capture_wait_ms,cpu_submit_ms,gpu_queries_completed,gpu_queries_dropped,preset,power,split\n";
        sessionStart=Clock::now();
    }
    void Resize() {
        if (!swap) return;
        RECT r{};
        GetClientRect(hwnd,&r);
        width=r.right-r.left; height=r.bottom-r.top;
        if (!width || !height) return;
        context->OMSetRenderTargets(0,nullptr,nullptr);
        target.Reset();
        Check(swap->ResizeBuffers(0,width,height,DXGI_FORMAT_UNKNOWN,0),"ResizeBuffers");
        ComPtr<ID3D11Texture2D> back;
        Check(swap->GetBuffer(0,IID_PPV_ARGS(&back)),"GetBuffer");
        Check(device->CreateRenderTargetView(back.Get(),nullptr,&target),"RTV");
    }
    void ResetCapture() {
        context->Flush();
        duplication.Reset(); capture.Reset(); input.Reset();
        Check(output1->DuplicateOutput(device.Get(),&duplication),"DuplicateOutput");
    }
    void Tick() {
        if (!duplication || !target) return;
        DXGI_OUTDUPL_FRAME_INFO info{};
        ComPtr<IDXGIResource> frame;
        auto start=Clock::now();
        HRESULT hr=duplication->AcquireNextFrame(8,&info,&frame);
        waitMs=std::chrono::duration<float,std::milli>(Clock::now()-start).count();
        if (hr==DXGI_ERROR_WAIT_TIMEOUT) { Sleep(1); return; }
        if (hr==DXGI_ERROR_ACCESS_LOST || hr==DXGI_ERROR_INVALID_CALL) {
            Sleep(150); ResetCapture(); return;
        }
        Check(hr,"AcquireNextFrame");
        try {
            ComPtr<ID3D11Texture2D> src;
            Check(frame.As(&src),"captured texture");
            D3D11_TEXTURE2D_DESC desc{};
            src->GetDesc(&desc);
            if (!capture || desc.Width!=sourceW || desc.Height!=sourceH) {
                capture.Reset();input.Reset();
                desc.Usage=D3D11_USAGE_DEFAULT;
                desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
                desc.CPUAccessFlags=0;desc.MiscFlags=0;
                desc.MipLevels=1;desc.ArraySize=1;
                Check(device->CreateTexture2D(&desc,nullptr,&capture),"GPU capture texture");
                Check(device->CreateShaderResourceView(capture.Get(),nullptr,&input),"SRV");
                sourceW=desc.Width;sourceH=desc.Height;
            }
            auto submitted=Clock::now();
            context->CopyResource(capture.Get(),src.Get());
            context->OMSetRenderTargets(1,target.GetAddressOf(),nullptr);
            float clear[4]={.04f,.045f,.06f,1};
            context->ClearRenderTargetView(target.Get(),clear);
            float ar=(float)sourceW/sourceH,windowAr=(float)width/height;
            D3D11_VIEWPORT viewport{};
            if (windowAr>ar) {
                viewport.Height=(float)height;
                viewport.Width=viewport.Height*ar;
                viewport.TopLeftX=(width-viewport.Width)*.5f;
            } else {
                viewport.Width=(float)width;
                viewport.Height=viewport.Width/ar;
                viewport.TopLeftY=(height-viewport.Height)*.5f;
            }
            viewport.MaxDepth=1;
            context->RSSetViewports(1,&viewport);
            Controls state;
            state.texel[0]=1.0f/sourceW;state.texel[1]=1.0f/sourceH;
            state.power=power;state.split=split?1.0f:0.0f;
            state.settings[0]=(float)preset;
            context->UpdateSubresource(uniforms.Get(),0,nullptr,&state,0,0);
            context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            context->VSSetShader(vs.Get(),nullptr,0);
            context->PSSetShader(ps.Get(),nullptr,0);
            context->PSSetConstantBuffers(0,1,uniforms.GetAddressOf());
            context->PSSetShaderResources(0,1,input.GetAddressOf());
            context->PSSetSamplers(0,1,sampler.GetAddressOf());
            bool measuring=gpuTimer->Begin(context.Get());
            context->Draw(3,0);
            if(measuring) gpuTimer->End(context.Get());
            ID3D11ShaderResourceView* empty[]={nullptr};
            context->PSSetShaderResources(0,1,empty);
            submitMs=std::chrono::duration<float,std::milli>(Clock::now()-submitted).count();
        } catch (...) {
            duplication->ReleaseFrame();
            throw;
        }
        Check(duplication->ReleaseFrame(),"ReleaseFrame");
        hr=swap->Present(vsync?1:0,0);
        if (hr==DXGI_STATUS_OCCLUDED) { Sleep(20); return; }
        Check(hr,"Present");
        gpuTimer->Poll(context.Get());
        ++frames;
        auto now=Clock::now();
        double seconds=std::chrono::duration<double>(now-lastTitle).count();
        if (seconds>=1.0) {
            wchar_t title[360]{};
            float fps=static_cast<float>(frames/seconds);
            std::swprintf(title,sizeof(title)/sizeof(title[0]),
                L"GameVision | %.1f capture FPS | GPU shader %.3f ms | wait %.2f ms | CPU submit %.2f ms | preset %d | %.1fx | split %s | vsync %s",
                fps,gpuTimer->LatestMs(),waitMs,submitMs,preset,power,
                split?L"on":L"off",vsync?L"on":L"off");
            if (csv.is_open()) {
                csv << std::fixed << std::setprecision(4)
                    << std::chrono::duration<double>(now-sessionStart).count() << ","
                    << fps << "," << gpuTimer->LatestMs() << ","
                    << waitMs << "," << submitMs << ","
                    << gpuTimer->Completed() << "," << gpuTimer->Dropped() << ","
                    << preset << "," << power << "," << (split?1:0) << "\n";
                csv.flush();
            }
            SetWindowTextW(hwnd,title);
            frames=0;lastTitle=now;
        }
    }
private:
    HWND hwnd=nullptr;
    std::unique_ptr<GpuTimer> gpuTimer;
    std::ofstream csv;
    Clock::time_point sessionStart=Clock::now();
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<IDXGIOutput1> output1;
    ComPtr<IDXGIOutputDuplication> duplication;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGISwapChain1> swap;
    ComPtr<ID3D11RenderTargetView> target;
    ComPtr<ID3D11Texture2D> capture;
    ComPtr<ID3D11ShaderResourceView> input;
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11Buffer> uniforms;
    UINT width=0,height=0,sourceW=0,sourceH=0;
    float waitMs=0,submitMs=0;
    unsigned frames=0;
    Clock::time_point lastTitle=Clock::now();
};
static GameVision* g=nullptr;
static LRESULT CALLBACK Proc(HWND wnd,UINT msg,WPARAM key,LPARAM lp) {
    if (msg==WM_DESTROY) { PostQuitMessage(0); return 0; }
    if (msg==WM_KEYDOWN && g) {
        switch (key) {
        case VK_ESCAPE: DestroyWindow(wnd);return 0;
        case '1':case '2':case '3':case '4': g->preset=(int)key-'1';return 0;
        case VK_SPACE:g->split=!g->split;return 0;
        case 'V':g->vsync=!g->vsync;return 0;
        case VK_OEM_PLUS:case VK_ADD:g->power=std::min(2.0f,g->power+.1f);return 0;
        case VK_OEM_MINUS:case VK_SUBTRACT:g->power=std::max(0.0f,g->power-.1f);return 0;
        }
    }
    return DefWindowProcW(wnd,msg,key,lp);
}
static int Run(HINSTANCE instance,int argc,wchar_t** argv) {
    unsigned monitor=0;
    std::wstring csvPath;
    for (int i=1;i<argc;++i) {
        if (!std::wcscmp(argv[i],L"--help")) {
            MessageBoxW(nullptr,L"GameVision.exe [--monitor N] [--csv benchmark.csv]\n1-4 presets, Space compare, +/- strength, V vsync, Esc quit.",
                L"GameVision help",MB_OK);
            return 0;
        }
        if (!std::wcscmp(argv[i],L"--csv") && i+1<argc) {
            csvPath=argv[++i];
            if (csvPath.empty()) throw std::runtime_error("Empty --csv output path.");
        } else if (!std::wcscmp(argv[i],L"--monitor") && i+1<argc) {
            wchar_t* end=nullptr;
            auto n=std::wcstoul(argv[++i],&end,10);
            if (end==argv[i] || *end || n>31) throw std::runtime_error("Invalid --monitor (0..31)");
            monitor=(unsigned)n;
        } else throw std::runtime_error("Unknown argument. Use --help.");
    }
    WNDCLASSEXW cls{};
    cls.cbSize=sizeof(cls);cls.lpfnWndProc=Proc;cls.hInstance=instance;
    cls.lpszClassName=L"GameVisionPreview";
    cls.hCursor=LoadCursor(nullptr,IDC_ARROW);
    cls.hbrBackground=(HBRUSH)GetStockObject(BLACK_BRUSH);
    if (!RegisterClassExW(&cls)) throw std::runtime_error("RegisterClassExW failed");
    HWND wnd=CreateWindowExW(0,cls.lpszClassName,L"GameVision | GPU preview",
        WS_OVERLAPPEDWINDOW|WS_VISIBLE,CW_USEDEFAULT,CW_USEDEFAULT,1280,760,
        nullptr,nullptr,instance,nullptr);
    if (!wnd) throw std::runtime_error("CreateWindowExW failed");
    // Windows 10 2004+: best-effort exclusion of own window from capture.
    SetWindowDisplayAffinity(wnd,0x00000011);
    GameVision game;
    g=&game;game.Initialize(wnd,monitor);
    if (!csvPath.empty()) game.OpenCsv(csvPath);
    RECT prev{};GetClientRect(wnd,&prev);
    MSG msg{};
    bool done=false;
    while (!done) {
        while (PeekMessageW(&msg,nullptr,0,0,PM_REMOVE)) {
            if (msg.message==WM_QUIT) { done=true;break; }
            TranslateMessage(&msg);DispatchMessageW(&msg);
        }
        if (done) break;
        RECT current{};GetClientRect(wnd,&current);
        if ((current.right!=prev.right || current.bottom!=prev.bottom)
            && current.right>0 && current.bottom>0) {
            game.Resize();prev=current;
        }
        if (IsIconic(wnd)) { Sleep(40);continue; }
        game.Tick();
    }
    g=nullptr;
    return 0;
}
int WINAPI wWinMain(HINSTANCE inst,HINSTANCE,PWSTR,int) {
    try {
        int argc=0;
        wchar_t** argv=CommandLineToArgvW(GetCommandLineW(),&argc);
        if (!argv) throw std::runtime_error("CommandLineToArgvW failed");
        int result=0;
        try { result=Run(inst,argc,argv); }
        catch (...) { LocalFree(argv);throw; }
        LocalFree(argv);
        return result;
    } catch (const std::exception& e) {
        MessageBoxA(nullptr,e.what(),"GameVision error",MB_OK|MB_ICONERROR);
        return 1;
    }
}