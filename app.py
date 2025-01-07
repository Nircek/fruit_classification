import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GdkPixbuf


settings = Gtk.Settings.get_default()
settings.set_property("gtk-theme-name", "Numix")
settings.set_property("gtk-application-prefer-dark-theme", False)

class FruitClassificationApp(Gtk.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app)
        self.set_title("Fruit Classification")
        self.set_default_size(400, 400)

        # Main vertical box layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_child(vbox)

        # Top Bar (Header)
        top_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        vbox.append(top_bar)

        # Left Button
        left_btn = Gtk.Button(label="\N{LEFTWARDS BLACK ARROW}")
        left_btn.set_size_request(50, 50)
        top_bar.append(left_btn)

        # Plus Button
        plus_btn = Gtk.Button(label="\N{HEAVY PLUS SIGN}")
        plus_btn.set_size_request(50, 50)
        top_bar.append(plus_btn)

        # Center Label (1/20)
        label = Gtk.Label(label="1/20")
        label.set_hexpand(True)
        label.set_justify(Gtk.Justification.CENTER)
        top_bar.append(label)

        # Up Button
        camera_btn = Gtk.Button(label="\N{CAMERA}")
        camera_btn.set_size_request(50, 50)
        top_bar.append(camera_btn)

        # Right Button
        right_btn = Gtk.Button(label="\N{RIGHTWARDS BLACK ARROW}")
        right_btn.set_size_request(50, 50)
        top_bar.append(right_btn)

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.append(separator)

        # Center Image
        self.image = Gtk.Picture()
        self.image.set_hexpand(True)
        self.image.set_vexpand(True)
        self.image.set_content_fit(Gtk.ContentFit.CONTAIN)
        vbox.append(self.image)

        # Load and display image
        self.load_image("sand.jpg")

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        vbox.append(separator)

        # Bottom Label
        bottom_label = Gtk.Label(label="APPLE")
        bottom_label.set_hexpand(True)
        bottom_label.set_justify(Gtk.Justification.CENTER)
        bottom_label.set_margin_top(10)
        bottom_label.set_margin_bottom(10)
        vbox.append(bottom_label)

    def load_image(self, path):
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                path, width=-1, height=-1, preserve_aspect_ratio=True
            )
            self.image.set_pixbuf(pixbuf)
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

    def do_activate(self):
        win = FruitClassificationApp(self)
        win.present()

app = FruitClassificationApplication()
app.run()
