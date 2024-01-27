use std::default::Default;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use winit::{
    event::*,
    event_loop::{EventLoop},
    window::WindowBuilder,
    window::Window,
};
use winit::dpi::PhysicalSize;
use winit::keyboard::{Key, NamedKey};

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    window: Window,
}

impl State {
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        // The instance is used to create the surface and the adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // the part of the window that we draw to
        let surface = unsafe {
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(&window).unwrap())
        }.unwrap();
        // The adapter is the handle for the GPU
        // use it to get info about the gpu (name, which backend, etc)
        // use it to create Device and Queue later
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptionsBase {
                // LowPower or HighPerformance -> LP picks adapter that favors battery life (integrated GPU)
                power_preference: wgpu::PowerPreference::default(),
                // tells wgpu to find an adapter that can present to the supplied surface
                compatible_surface: Some(&surface),
                // forces wgpu to pick an adapter that will work on all hardware
                // rendering backend will use a "software" system instead of GPU hardware
                force_fallback_adapter: false,
            }
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                label: None,
            },
            None,
        ).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps.formats.iter()
            .copied()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width, //TODO: ensure these aren't 0
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            desired_maximum_frame_latency: 1,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn scale(&mut self, scale_factor: &f64) {
        self.size = PhysicalSize::new(self.config.width * (*scale_factor) as u32, self.config.height * (*scale_factor) as u32)
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {
        // todo!()
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder")
        });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0
                        }),
                        store: wgpu::StoreOp::Store
                    }
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None
            });
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())

    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't init logger");
        } else {
            env_logger::init();
        }
    }
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let mut state = State::new(window).await;

    event_loop.run(move |event, control_flow| {
        match event {
            Event::AboutToWait => {
                state.window().request_redraw();
            }

            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state.window.id() => if !state.input(event) {
                match event {
                    WindowEvent::RedrawRequested => {
                        state.update();
                        match state.render() {
                            Ok(_) => {}

                            //Reconfig the surface if lost
                            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                            //System out of memory, quit program
                            Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
                            Err(e) => eprintln!("{:?}", e)
                        }
                    }

                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size)
                    }

                    WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                        state.scale(scale_factor)
                    }

                    WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            state: ElementState::Pressed,
                            logical_key: Key::Named(NamedKey::Escape),
                            ..
                        },
                        ..
                    } => control_flow.exit(),
                    _ => {}
                }
            }
            _ => {}
        }
    }).expect("TODO: panic message");

    #[cfg(target_arch = "wasm32")]
    {
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            }).expect("Couldn't append canvas to document body :(");
    }
}