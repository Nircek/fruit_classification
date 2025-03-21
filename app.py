# flake8: noqa: E402
# pylint: disable=C0114, C0116, C0115, C0413, E1101
from time import sleep
import tempfile
from pathlib import Path
import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
from PIL import Image, ImageOps
from cv2 import VideoCapture, imwrite

model_loading_dialog = Gtk.MessageDialog(
    title="Fruit Classification",
    flags=0,
    message_type=Gtk.MessageType.INFO,
    buttons=Gtk.ButtonsType.OK,
    text="Loading the model...",
)
model_loading_dialog.show()

while Gtk.events_pending():
    Gtk.main_iteration()
    sleep(0.1)
from net import predict

model_loading_dialog.destroy()


settings = Gtk.Settings.get_default()
settings.set_property("gtk-application-prefer-dark-theme", False)


class FruitClassificationApp(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app)
        self.set_title("Fruit Classification")
        self.set_default_size(400, 400)

        self.images = [self.load_pixbuf("sand.jpg")]
        self.predictions = ["sand"]
        self.image_index = 0
        self.broken = self.load_pixbuf("broken.jpg")

        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            b"""
        .arrow {
            font-size: 36px;
        }
        .button {
            font-size: 28px;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
        }
        """
        )
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(vbox)

        top_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        vbox.pack_start(top_bar, False, False, 0)

        previous_btn = Gtk.Button(label="\N{LEFTWARDS BLACK ARROW}")
        previous_btn.set_size_request(80, 80)
        previous_btn.get_style_context().add_class("arrow")
        previous_btn.connect("clicked", self.go_to_previous)
        top_bar.pack_start(previous_btn, False, False, 0)

        add_btn = Gtk.Button(label="\N{HEAVY PLUS SIGN}")
        add_btn.set_size_request(80, 80)
        add_btn.get_style_context().add_class("button")
        add_btn.connect("clicked", self.go_to_add)
        top_bar.pack_start(add_btn, False, False, 0)

        self.image_label = Gtk.Label()
        self.image_label.set_hexpand(True)
        self.image_label.set_justify(Gtk.Justification.CENTER)
        top_bar.pack_start(self.image_label, False, False, 0)

        take_picture_btn = Gtk.Button(label="\N{CAMERA}")
        take_picture_btn.set_size_request(80, 80)
        take_picture_btn.get_style_context().add_class("button")
        take_picture_btn.connect("clicked", self.go_to_take_picture)
        top_bar.pack_start(take_picture_btn, False, False, 0)

        next_btn = Gtk.Button(label="\N{RIGHTWARDS BLACK ARROW}")
        next_btn.set_size_request(80, 80)
        next_btn.get_style_context().add_class("arrow")
        next_btn.connect("clicked", self.go_to_next)
        top_bar.pack_start(next_btn, False, False, 0)

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.pack_start(separator, False, False, 0)

        self.image = Gtk.Image()
        self.image.set_hexpand(True)
        self.image.set_vexpand(True)

        self.image.set_halign(Gtk.Align.CENTER)
        self.image.set_valign(Gtk.Align.CENTER)
        vbox.pack_start(self.image, True, True, 0)

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.pack_start(separator, False, False, 0)

        self.predict_label = Gtk.Label()
        self.predict_label.set_hexpand(True)
        self.predict_label.set_justify(Gtk.Justification.CENTER)
        self.predict_label.set_margin_top(10)
        self.predict_label.set_margin_bottom(10)
        self.predict_label.get_style_context().add_class("prediction")
        vbox.pack_start(self.predict_label, False, False, 0)

        self.update_layout()
        self.show_all()
        self.set_resizable(False)

    def load_pixbuf(self, path):
        try:
            im = Image.open(path)
            if im.mode in ("RGBA", "LA") or (
                im.mode == "P" and "transparency" in im.info
            ):
                background = Image.new("RGB", im.size, (255, 255, 255))
                background.paste(
                    im, mask=im.getchannel("A") if im.mode == "RGBA" else None
                )
                im = background
            else:
                im = im.convert("RGB")
            im = ImageOps.contain(im, (400, 300))

            # SRC: https://gist.github.com/mozbugbox/10cd35b2872628246140
            data = im.tobytes()
            w, h = im.size
            data = GLib.Bytes.new(data)
            return GdkPixbuf.Pixbuf.new_from_bytes(
                data, GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
            )
            # return GdkPixbuf.Pixbuf.new_from_file_at_scale(
            #     path, width=400, height=300, preserve_aspect_ratio=True
            # )
        except gi.repository.GLib.GError as e:
            dialog = Gtk.MessageDialog(
                transient_for=self,
                flags=0,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text="Error loading image",
            )
            dialog.format_secondary_text(str(e))
            dialog.run()
            dialog.destroy()
        return None

    def set_pixbuf(self, pixbuf):
        self.image.set_from_pixbuf(self.broken if pixbuf is None else pixbuf)

    def update_layout(self):
        self.image_label.set_text(f"{self.image_index + 1}/{len(self.images)}")
        self.predict_label.set_text(self.predictions[self.image_index])
        self.set_pixbuf(self.images[self.image_index])

    def go_to_next(self, _ev):
        if self.image_index < len(self.images) - 1:
            self.image_index += 1
        self.update_layout()

    def go_to_previous(self, _ev):
        if self.image_index > 0:
            self.image_index -= 1
        self.update_layout()

    def go_to_take_picture(self, _ev):
        cam = VideoCapture(0)
        s, img = cam.read()
        tempdir = Path(tempfile.gettempdir())
        tempfilename = str(tempdir / "webcamera_shot.jpg")
        if s:
            imwrite(tempfilename, img)
            self.image_index = len(self.images)
            self.images.append(self.load_pixbuf(tempfilename))
            self.predictions.append(predict(tempfilename))
            self.update_layout()
        cam.release()

    def go_to_add(self, _ev):
        dialog = Gtk.FileChooserDialog(
            title="Please choose files to open",
            transient_for=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN,
            Gtk.ResponseType.OK,
        )
        dialog.set_select_multiple(True)

        response = dialog.run()
        paths = dialog.get_filenames() if response == Gtk.ResponseType.OK else []
        dialog.destroy()
        if not paths:
            return
        self.image_index = len(self.images)  # go to the first new image
        for path in paths:
            self.images.append(self.load_pixbuf(path))
            self.predictions.append(predict(path))
        self.update_layout()


class FruitClassificationApplication(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="io.github.nircek.fruit_classification")

    def do_activate(self, *args, **kwargs):
        win = FruitClassificationApp(self)
        win.present()


def main():
    app = FruitClassificationApplication()
    app.run()


if __name__ == "__main__":
    main()
