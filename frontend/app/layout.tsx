import type { Metadata } from 'next';
import '../styles/globals.css';

export const metadata: Metadata = {
  title: 'Multimodal Retrieval System',
  description: 'Production-ready multimodal search system for face, image, and text queries',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-void antialiased">
        <div className="fixed inset-0 grid-pattern pointer-events-none opacity-50" />
        <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-electric/5 rounded-full blur-3xl pointer-events-none" />
        <div className="fixed bottom-0 right-0 w-[600px] h-[400px] bg-neon-purple/5 rounded-full blur-3xl pointer-events-none" />
        <main className="relative z-10">
          {children}
        </main>
      </body>
    </html>
  );
}
