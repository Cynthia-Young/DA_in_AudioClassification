import os
import wx
import wx.media
import torch
from network import Transfer_Cnn14
from data_load import feature
import torch.nn as nn


idOPEN = wx.ID_OPEN
idPLAY = wx.NewId()
idTEST = wx.NewId()
idSTOP = wx.NewId()
idEXIT = wx.ID_EXIT

#-------------------------------------------------------------------------------------------
class AudioWindow(wx.Window):
    def __init__(self, parent, ID):
        wx.Window.__init__(self, parent, ID, style=wx.NO_FULL_REPAINT_ON_RESIZE)
        self.SetBackgroundColour("GREY")
        self.filename = ""
        self.media_ctrl = wx.media.MediaCtrl(self, style=wx.SIMPLE_BORDER)
        
        self.InitBuffer()
        
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_IDLE, self.OnIdle)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

        self.media_ctrl.Bind(wx.media.EVT_MEDIA_LOADED, self.OnMediaLoaded)

    def InitBuffer(self):
        size = self.GetClientSize()
        self.buffer = wx.Bitmap(max(1, size.width), max(1, size.height))
        dc = wx.BufferedDC(None, self.buffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        self.reInitBuffer = False

    def SetFilename(self, fn):
        self.filename = fn
        self.Refresh()

    def GetFilename(self):
        return self.filename

    def OnSize(self, event):
        self.reInitBuffer = True

    def OnIdle(self, event):
        if self.reInitBuffer:
            self.InitBuffer()
            self.Refresh(False)

    def OnPaint(self, event):
        self.InitBuffer()
        dc = wx.BufferedPaintDC(self, self.buffer)
        if self.filename:
            dc.DrawText(f"Loaded file: {self.filename}", 10, 10)

    def LoadAudioFile(self, filename):
        if filename:
            self.filename = filename
            self.media_ctrl.Load(filename)

    def OnMediaLoaded(self, event):
        # 在音频加载完成后自动播放
        self.media_ctrl.Play()

#-------------------------------------------------------------------------------------------
class ControlPanel(wx.Panel):
    def __init__(self, parent, ID, audio_win):
        wx.Panel.__init__(self, parent, ID, style=wx.RAISED_BORDER)
        self.audio_win = audio_win

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.buttons = []
        self.buttons.append(wx.Button(self, idOPEN, "Open Audio File"))
        self.buttons.append(wx.Button(self, idPLAY, "Play Recording"))
        self.buttons.append(wx.Button(self, idSTOP, "Stop Playback"))
        self.buttons.append(wx.Button(self, idTEST, "Test the Accent"))
        for btn in self.buttons:
            self.sizer.Add(btn, 1, wx.EXPAND)

        self.Bind(wx.EVT_BUTTON, self.OnOpen, id=idOPEN)
        self.Bind(wx.EVT_BUTTON, self.OnPlay, id=idPLAY)
        self.Bind(wx.EVT_BUTTON, self.OnStop, id=idSTOP)
        self.Bind(wx.EVT_BUTTON, self.OnTest, id=idTEST)
        
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(self.sizer, 0, wx.ALL)
        self.SetSizer(box)
        self.SetAutoLayout(True)
        box.Fit(self)

    def OnOpen(self, event):
        dlg = wx.FileDialog(self, "Open audio file...", os.getcwd(),
                            style=wx.FD_OPEN,
                            wildcard = "Audio files (*.wav)|*.wav")
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetPath()
            self.audio_win.SetFilename(self.filename)
            self.audio_win.media_ctrl.Load(self.filename)
        dlg.Destroy()

    def OnPlay(self, event):
        if self.audio_win.filename:
            self.audio_win.media_ctrl.Play()

    def OnStop(self, event):
        if self.audio_win.filename:
            self.audio_win.media_ctrl.Stop()

    def OnTest(self, event):
        fn = self.audio_win.GetFilename()
        if not fn:
            wx.MessageBox("Please open an audio file before testing!",
                          "Error", style=wx.OK|wx.ICON_EXCLAMATION)
            return

        # 使用模型进行口音分类
        aud = feature(fn, sample_rate=32000)
        aud = torch.from_numpy(aud).unsqueeze(0)
        
        netG = Transfer_Cnn14(sample_rate = 32000, window_size = 1024, hop_size= 320, mel_bins = 64, fmin = 50, fmax = 14000, 
        classes_num = 3, freeze_base = False, freeze_classifier = False)
        model = nn.Sequential(netG)
        model.load_state_dict(torch.load("model\m2s_ps_0.0_par_0.3final.pt", map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            prediction, _ = model(aud)
            predicted_label = torch.argmax(prediction, dim=1).item()

        # 根据 predicted_label 的值输出相应的口音信息
        if predicted_label == 0:
            accent = "USA"
        elif predicted_label == 1:
            accent = "India"
        elif predicted_label == 2:
            accent = "UK"
        else:
            accent = "Unknown"

        wx.MessageBox(f"The accent of the audio file is: {accent}",
                    "Test Result", style=wx.OK|wx.ICON_INFORMATION)

#-------------------------------------------------------------------------------------------
class AccentFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, "Accent Classification Test", size=(638,512),
                          style=wx.DEFAULT_FRAME_STYLE | wx.NO_FULL_REPAINT_ON_RESIZE)

        self.audio_win = AudioWindow(self, -1)           # 创建音频窗口对象
        cPanel = ControlPanel(self, -1, self.audio_win)  # 创建控制面板对象

        # 设置布局
        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(cPanel, 0, wx.EXPAND)
        box.Add(self.audio_win, 1, wx.EXPAND)

        self.SetSizer(box)
        self.Centre()  # 使UI居中显示

#-------------------------------------------------------------------------------------------
class AccentApp(wx.App):
    def OnInit(self):
        frame = AccentFrame(None)
        frame.Show(True)
        return True
    
#-------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app = AccentApp()
    app.MainLoop()
