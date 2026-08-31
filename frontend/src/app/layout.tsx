import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Foveated LiDAR 2.5D Mapping Dashboard',
  description: 'Real-time multi-resolution foveated LiDAR perception & semantic elevation mapping platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-foreground antialiased">{children}</body>
    </html>
  );
}
