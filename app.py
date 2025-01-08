# flake8: noqa: E402
# pylint: disable=C0114, C0116, C0115, C0413, E1101
import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GdkPixbuf


settings = Gtk.Settings.get_default()
settings.set_property("gtk-application-prefer-dark-theme", False)


class FruitClassificationApp(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app)
        self.set_title("Fruit Classification")
        self.set_default_size(400, 400)

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

        left_btn = Gtk.Button(label="\N{LEFTWARDS BLACK ARROW}")
        left_btn.set_size_request(80, 80)
        left_btn.get_style_context().add_class("arrow")
        top_bar.pack_start(left_btn, False, False, 0)

        plus_btn = Gtk.Button(label="\N{HEAVY PLUS SIGN}")
        plus_btn.set_size_request(80, 80)
        plus_btn.get_style_context().add_class("button")
        top_bar.pack_start(plus_btn, False, False, 0)

        label = Gtk.Label(label="1/20")
        label.set_hexpand(True)
        label.set_justify(Gtk.Justification.CENTER)
        top_bar.pack_start(label, False, False, 0)

        camera_btn = Gtk.Button(label="\N{CAMERA}")
        camera_btn.set_size_request(80, 80)
        camera_btn.get_style_context().add_class("button")
        camera_btn.set_sensitive(False)
        top_bar.pack_start(camera_btn, False, False, 0)

        right_btn = Gtk.Button(label="\N{RIGHTWARDS BLACK ARROW}")
        right_btn.set_size_request(80, 80)
        right_btn.get_style_context().add_class("arrow")
        top_bar.pack_start(right_btn, False, False, 0)

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.pack_start(separator, False, False, 0)

        self.image = Gtk.Image()
        self.image.set_hexpand(True)
        self.image.set_vexpand(True)

        self.image.set_halign(Gtk.Align.CENTER)
        self.image.set_valign(Gtk.Align.CENTER)
        vbox.pack_start(self.image, True, True, 0)

        self.load_image("sand.jpg")

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.pack_start(separator, False, False, 0)

        bottom_label = Gtk.Label(label="APPLE")
        bottom_label.set_hexpand(True)
        bottom_label.set_justify(Gtk.Justification.CENTER)
        bottom_label.set_margin_top(10)
        bottom_label.set_margin_bottom(10)
        bottom_label.get_style_context().add_class("prediction")
        vbox.pack_start(bottom_label, False, False, 0)

        self.show_all()
        self.set_resizable(False)

    def load_image(self, path):
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                path, width=400, height=300, preserve_aspect_ratio=True
            )
            self.image.set_from_pixbuf(pixbuf)
        except gi.repository.GLib.GError as e:
            dialog = Gtk.AlertDialog()
            dialog.set_message("Error loading image")
            dialog.set_detail(str(e))
            dialog.set_modal(True)
            dialog.set_buttons(["OK"])
            dialog.choose(self)


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
