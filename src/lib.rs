use std::default::Default;
use wgpu::util::DeviceExt;
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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3]
}

unsafe impl bytemuck::Pod for Vertex {}
unsafe impl bytemuck::Zeroable for Vertex {}


const VERTICES: &[Vertex] = &[
    Vertex { position: [0.0, 0.0, 0.0], color: [0.5, 0.0, 0.5] },
    Vertex { position: [0.0, -0.25, 0.0], color: [0.5, 0.0, 0.5] },
    Vertex { position: [-0.25, -0.25, 0.0], color: [0.5, 0.0, 0.5] },
    Vertex { position: [-0.25, 0.0, 0.0], color: [0.5, 0.0, 0.5] },
    Vertex { position: [-0.0833, 0.125, 0.0], color: [0.0, 0.5, 0.5] },
    Vertex { position: [0.125, 0.125, 0.0], color: [0.0, 0.5, 0.5] },
    Vertex { position: [0.125, -0.0833, 0.0], color: [0.25, 0.75, 0.0] },
];

const INDICES: &[u16] = &[
    0,2,1,
    2,0,3,
    3,0,4,
    4,0,5,
    0,6,5,
    0,1,6
];

const TRANSLATE_Y: f32 = 0.25;
const TRANSLATE_X: f32 = 0.0;

const VERTICES2: &[Vertex] = &[
    Vertex { position: [0.0+ TRANSLATE_X, 0.0+ TRANSLATE_Y, 0.0], color: [0.5, 0.0, 0.5] },
    Vertex { position: [0.0+ TRANSLATE_X, -0.25+ TRANSLATE_Y, 0.0], color: [0.5, 0.0, 0.5] },
    Vertex { position: [-0.25+ TRANSLATE_X, -0.25+ TRANSLATE_Y, 0.0], color: [0.5, 0.0, 0.5] },
    Vertex { position: [-0.25+ TRANSLATE_X, 0.0+ TRANSLATE_Y, 0.0], color: [0.5, 0.0, 0.5] },
    Vertex { position: [-0.0833+ TRANSLATE_X, 0.125+ TRANSLATE_Y, 0.0], color: [0.0, 0.5, 0.5] },
    Vertex { position: [0.125+ TRANSLATE_X, 0.125+ TRANSLATE_Y, 0.0], color: [0.0, 0.5, 0.5] },
    Vertex { position: [0.125+ TRANSLATE_X, -0.0833+ TRANSLATE_Y, 0.0], color: [0.25, 0.75, 0.0] },
];

const INDICES2: &[u16] = &[
    0,2,1,
    2,0,3,
    3,0,4,
    4,0,5,
    0,6,5,
    0,1,6
];

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES
        }
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    clear_color: wgpu::Color,
    shape_toggle: bool,
    window: Window,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    vertex_buffer2: wgpu::Buffer,
    index_buffer2: wgpu::Buffer,
    num_indices2: u32
}

impl State {
    async fn new(window: Window) -> Self {
        let shape_toggle = true;
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

        let clear_color = wgpu::Color::BLACK;

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[]
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()]
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL
                })]
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None
        });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX
            }
        );

        let vertex_buffer2 = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES2),
                usage: wgpu::BufferUsages::VERTEX
            }
        );

        let index_buffer2 = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES2),
                usage: wgpu::BufferUsages::INDEX
            }
        );


        let num_indices = INDICES.len() as u32;
        let num_indices2 = INDICES2.len() as u32;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
            shape_toggle,
            window,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            vertex_buffer2,
            index_buffer2,
            num_indices2
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
        match event {
            WindowEvent::KeyboardInput { event: KeyEvent {
                state: ElementState::Pressed,
                logical_key: Key::Named(NamedKey::Space),
                ..
            },
                ..
            } => {
                self.shape_toggle = !self.shape_toggle;
                true
            },
            WindowEvent::CursorMoved {position, .. } => {
                self.clear_color = wgpu::Color {
                    r: position.x / self.size.width as f64,
                    g: position.y / self.size.height as f64,
                    b: 0.5,
                    a: 1.0
                };
                true
            }
            _ => false
        }
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
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store
                    }
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None
            });

            render_pass.set_pipeline(&self.render_pipeline);
            if !self.shape_toggle {
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices, 0,0..1);
            } else {
                render_pass.set_vertex_buffer(0, self.vertex_buffer2.slice(..));
                render_pass.set_index_buffer(self.index_buffer2.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..self.num_indices2, 0,0..1);
            }
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