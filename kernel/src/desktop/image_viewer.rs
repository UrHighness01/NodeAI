//! Image Viewer — Phase 26. Supports BMP (24-bit uncompressed) and raw RGBA.
//!
//! Features:
//!  - Load a file from VFS (BMP or raw RGBA)
//!  - Display scaled/centered in window
//!  - +/- to zoom, arrow keys to pan

use alloc::vec::Vec;
use alloc::string::String;
use spin::{Mutex, Once};
use crate::desktop::compositor::{
    wm_create_window, wm_fill_window_rect, wm_paint_pixel, wm_flip,
};

const WIN_W: u32 = 800;
const WIN_H: u32 = 600;
const BG:    u32 = 0xFF111111;

struct Image {
    width:  u32,
    height: u32,
    pixels: Vec<u32>,   // ARGB 0xFFRRGGBB
}

impl Image {
    fn from_bmp(data: &[u8]) -> Option<Self> {
        if data.len() < 54 { return None; }
        // BITMAPFILEHEADER
        if &data[0..2] != b"BM" { return None; }
        let pixel_offset = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
        // BITMAPINFOHEADER
        let width  = i32::from_le_bytes([data[18], data[19], data[20], data[21]]).unsigned_abs();
        let height = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);
        let flip   = height > 0;
        let height = height.unsigned_abs();
        let bpp    = u16::from_le_bytes([data[28], data[29]]);
        let compression = u32::from_le_bytes([data[30], data[31], data[32], data[33]]);
        if bpp != 24 || compression != 0 { return None; }
        let row_stride = ((width * 3 + 3) / 4) * 4;
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            let src_row = if flip { height - 1 - row } else { row };
            let off = pixel_offset + (src_row as usize) * row_stride as usize;
            for col in 0..width as usize {
                let b = off + col * 3;
                if b + 2 >= data.len() { pixels.push(0xFF000000); continue; }
                let blue  = data[b] as u32;
                let green = data[b+1] as u32;
                let red   = data[b+2] as u32;
                pixels.push(0xFF000000 | (red << 16) | (green << 8) | blue);
            }
        }
        Some(Self { width, height, pixels })
    }
}

struct ImageViewer {
    win_id: u32,
    image:  Option<Image>,
    filename: String,
    zoom:   f32,   // 1.0 = 1:1
    pan_x:  i32,
    pan_y:  i32,
}

static IMG_APP: Once<Mutex<ImageViewer>> = Once::new();

impl ImageViewer {
    fn render(&self) {
        let id = self.win_id;
        wm_fill_window_rect(id, 0, 0, WIN_W, WIN_H, BG);
        if let Some(img) = &self.image {
            let dw = ((img.width  as f32) * self.zoom) as u32;
            let dh = ((img.height as f32) * self.zoom) as u32;
            // Center
            let ox = (WIN_W as i32 - dw as i32) / 2 + self.pan_x;
            let oy = (WIN_H as i32 - dh as i32) / 2 + self.pan_y;
            for dy in 0..dh {
                for dx in 0..dw {
                    let px = ox + dx as i32;
                    let py = oy + dy as i32;
                    if px < 0 || py < 0 || px >= WIN_W as i32 || py >= WIN_H as i32 { continue; }
                    let sx = (dx as f32 / self.zoom) as u32;
                    let sy = (dy as f32 / self.zoom) as u32;
                    let idx = (sy * img.width + sx) as usize;
                    if idx < img.pixels.len() {
                        wm_paint_pixel(id, px as u32, py as u32, img.pixels[idx]);
                    }
                }
            }
        } else {
            // Show "no image" text
            use crate::desktop::compositor::wm_draw_text_cell;
            let msg = b"No image loaded. Use: imgview <filename>";
            for (i, &b) in msg.iter().enumerate() {
                wm_draw_text_cell(id, 20 + i as u32 * 8, WIN_H / 2, b, 0xFFAAAAAA, BG);
            }
        }
        wm_flip(id);
    }

    fn handle_key(&mut self, ch: u8) {
        match ch {
            b'+' | b'=' => { self.zoom = (self.zoom * 1.25).min(8.0); }
            b'-'        => { self.zoom = (self.zoom * 0.8).max(0.1); }
            b'0'        => { self.zoom = 1.0; self.pan_x = 0; self.pan_y = 0; }
            0x43 /* C */ => { self.pan_x += 16; } // right arrow
            0x44 /* D */ => { self.pan_x -= 16; } // left arrow
            0x41 /* A */ => { self.pan_y -= 16; } // up arrow
            0x42 /* B */ => { self.pan_y += 16; } // down arrow
            _ => {}
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn imgview_open(filename: &str) {
    let data: Vec<u8> = if !filename.is_empty() {
        crate::vfs::read_file(filename).unwrap_or_default()
    } else { Vec::new() };

    let image = if !data.is_empty() { Image::from_bmp(&data) } else { None };

    IMG_APP.call_once(|| {
        let id = wm_create_window(90, 60, WIN_W, WIN_H, "Image Viewer");
        Mutex::new(ImageViewer {
            win_id: id, image, filename: String::from(filename),
            zoom: 1.0, pan_x: 0, pan_y: 0,
        })
    });

    // If already open, reload file
    if let Some(app) = IMG_APP.get() {
        let mut g = app.lock();
        if !filename.is_empty() {
            let data2: Vec<u8> = crate::vfs::read_file(filename).unwrap_or_default();
            g.image = Image::from_bmp(&data2);
            g.filename = String::from(filename);
            g.zoom = 1.0; g.pan_x = 0; g.pan_y = 0;
        }
        g.render();
    }
}

pub fn imgview_is_open() -> bool { IMG_APP.get().is_some() }

pub fn imgview_key(ch: u8) {
    if let Some(app) = IMG_APP.get() {
        let mut g = app.lock();
        g.handle_key(ch);
        g.render();
    }
}
